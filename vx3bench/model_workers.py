"""Independent, repeat-inference processes for configured ONNX models.

Measurement design (why the numbers are precise)
------------------------------------------------
* Nothing crosses a process boundary per inference. Earlier versions pushed
  every single inference through a multiprocessing.Manager (a lock round-trip,
  two proxied list appends, and a full re-read/re-write of the timestamp list,
  all over IPC). That bookkeeping sat inside the inference loop, so it both
  slowed the loop down and was counted as "model time". Each model process
  now keeps its statistics in plain local variables and publishes a small
  JSON snapshot about once a second, exactly like the stream processes do.
* Only ``session.run`` is timed, with ``time.perf_counter``. For CUDA and
  TensorRT the input tensor is uploaded to the GPU once and bound with
  IOBinding, and outputs stay on the GPU, so each timed call is the model's
  own compute -- not a host->device copy of the same fixed image followed by
  a device->host copy of results nobody reads. Outputs are explicitly
  synchronized inside the timed region so asynchronous GPU work can never be
  under-reported.
* Warm-up inferences (CUDA context creation, cuDNN algorithm search, TensorRT
  engine build) are excluded from FPS and latency. The very first inference
  is still reported separately as ``cold_start_latency_ms``.
* FPS is measured over the model's own measurement window (from the end of
  warm-up to the last completed inference), not from when the parent process
  constructed the worker object -- so model loading and shutdown time no
  longer drag the average down.
* The few microseconds spent publishing the snapshot are timed and removed
  from the FPS window (``instrumentation_seconds``), so the reported rate is
  what the model loop achieves on its own.
* Latency percentiles come from a 1-microsecond-resolution histogram, so they
  stay exact to 1 us over a 24 h soak without storing every sample.

Pacing comes only from the scaler (each instance's share of the per-model
target). The old per-model ``inference_time`` idle gap has been removed: with
target pacing it was ignored, and it is silently ignored if an old scenario
still contains it. With no target (target_fps_per_stream = 0) instances run
back-to-back to measure maximum throughput.
"""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from .config import slug
from .live_publish import publish_json

DEVICE_CHOICES = ("auto", "cuda", "tensorrt", "directml", "coreml", "cpu")

# ONNX Runtime execution providers, by device choice, in the order they are
# TRIED. A provider is used only if a session really runs on it: a provider
# can be listed by ONNX Runtime without its libraries being installed (e.g.
# onnxruntime-gpu on a machine without CUDA/TensorRT), in which case ONNX
# Runtime silently runs on the CPU. Each step down is recorded and shown.
_GPU_CHAIN = ["CUDAExecutionProvider", "ROCMExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider"]
PROVIDER_CHAINS = {
    "tensorrt": ["TensorrtExecutionProvider"] + _GPU_CHAIN + ["CPUExecutionProvider"],
    "cuda": _GPU_CHAIN + ["CPUExecutionProvider"],
    "directml": ["DmlExecutionProvider", "CUDAExecutionProvider", "ROCMExecutionProvider", "CoreMLExecutionProvider",
                 "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    "auto": _GPU_CHAIN + ["CPUExecutionProvider"],
    "cpu": ["CPUExecutionProvider"],
}
PROVIDER_LABELS = {"TensorrtExecutionProvider": "GPU - TensorRT (NVIDIA)", "CUDAExecutionProvider": "GPU - CUDA (NVIDIA)",
                   "ROCMExecutionProvider": "GPU - ROCm (AMD)", "DmlExecutionProvider": "GPU - DirectML (Windows)",
                   "CoreMLExecutionProvider": "CoreML (Apple)", "CPUExecutionProvider": "CPU"}
DEFAULT_DEVICE = "auto"
DEFAULT_WARMUP = 10
_HIST_BINS = 1_000_000          # 1 us bins -> latencies up to 1 s are exact to 1 us
_PUBLISH_INTERVAL = 1.0         # seconds between live snapshots
_PERCENTILE_INTERVAL = 10.0     # live percentiles are refreshed less often (final is exact)
_FPS_WINDOW = 5.0               # rolling window for instant FPS


def normalize_device(value: Any) -> str:
    device = str(value if value is not None else DEFAULT_DEVICE).strip().lower()
    return "cuda" if device == "gpu" else device


def model_snapshot_path(out_dir, name) -> Path:
    return Path(out_dir) / "ai" / f"{slug(str(name))}.json"


# --------------------------------------------------------------------------- #
# Child-process helpers
# --------------------------------------------------------------------------- #
def _dimension(value: Any, default: int) -> int:
    return int(value) if isinstance(value, int) and value > 0 else default


def _input_tensor(np, shape: list[Any], input_type: str):
    """Build one reusable image tensor from the model's declared input shape."""
    if len(shape) != 4:
        raise ValueError(f"expected a 4-D image input, got {shape!r}")
    second = shape[1] if isinstance(shape[1], int) else None
    last = shape[3] if isinstance(shape[3], int) else None
    if second in (1, 3, 4):
        layout, channels = "NCHW", second
        height, width = _dimension(shape[2], 640), _dimension(shape[3], 640)
        tensor_shape = (1, channels, height, width)
    elif last in (1, 3, 4):
        layout, channels = "NHWC", last
        height, width = _dimension(shape[1], 640), _dimension(shape[2], 640)
        tensor_shape = (1, height, width, channels)
    else:
        layout, channels, height, width = "NCHW", 3, 640, 640
        tensor_shape = (1, channels, height, width)
    dtype = np.float16 if "float16" in input_type else np.float32
    # A fixed image is intentional: this worker measures sustained inference
    # contention, not capture, decode, or an input queue.
    return np.zeros(tensor_shape, dtype=dtype), layout, width, height


def _provider_config(name: str, cfg: dict[str, Any], model_path: str):
    """(providers list for InferenceSession, needs DirectML session tweaks)."""
    device_id = int(cfg.get("device_id", 0) or 0)
    cuda_options = {
        "device_id": device_id,
        "arena_extend_strategy": "kSameAsRequested",
        "gpu_mem_limit": str(4 * 1024 * 1024 * 1024),
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": "1",
        "cudnn_conv_use_max_workspace": "1",
    }
    if name == "TensorrtExecutionProvider":
        cache = Path(model_path).parent / ".trt_cache"
        try:
            cache.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        return [("TensorrtExecutionProvider", {"device_id": device_id, "trt_engine_cache_enable": True,
                                               "trt_engine_cache_path": str(cache)}),
                ("CUDAExecutionProvider", cuda_options)]
    if name == "CUDAExecutionProvider":
        return [("CUDAExecutionProvider", cuda_options)]
    if name in ("ROCMExecutionProvider", "DmlExecutionProvider"):
        return [(name, {"device_id": device_id})]
    return [(name, {})]


def _create_session(ort, device: str, cfg: dict[str, Any], model_path: str, base_options, info: dict[str, Any]):
    """Create the session on the best WORKING provider for `device`,
    stepping down the chain (ending with the CPU) when one is not installed
    or fails to initialise. Never raises for a missing GPU -- only if the
    model cannot run at all (e.g. a broken model file)."""
    available = set(ort.get_available_providers())
    chain = PROVIDER_CHAINS.get(device) or PROVIDER_CHAINS["auto"]
    steps = []
    last_exc = None
    for name in chain:
        if name not in available:
            steps.append(f"{PROVIDER_LABELS.get(name, name)}: not installed")
            continue
        options = base_options()
        if name == "DmlExecutionProvider":            # DirectML requirements
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if name != "CPUExecutionProvider":
            options.log_severity_level = 3            # failed GPU attempts are reported below, not spammed
        providers = _provider_config(name, cfg, model_path)
        if name != "CPUExecutionProvider":
            providers = providers + ["CPUExecutionProvider"]
        try:
            session = ort.InferenceSession(model_path, options, providers=providers)
        except Exception as exc:
            last_exc = exc
            steps.append(f"{PROVIDER_LABELS.get(name, name)}: failed to start ({str(exc).splitlines()[0][:160]})")
            continue
        active = session.get_providers()[0]
        if active != name:
            # Listed but not usable here (e.g. CUDA/TensorRT libraries or a
            # GPU missing): ONNX Runtime quietly ran on the CPU instead.
            steps.append(f"{PROVIDER_LABELS.get(name, name)}: not usable on this machine (ONNX Runtime fell back to "
                         f"{PROVIDER_LABELS.get(active, active)})")
            del session
            continue
        info["provider"] = active
        info["active_device"] = PROVIDER_LABELS.get(active, active)
        # Explain every step down from what was asked for ("auto" simply
        # means "best available", so for it the steps are informational).
        if steps:
            tried = [x for x in steps if "not installed" not in x]
            reason = "; ".join(tried) if tried else "no GPU provider is installed in this ONNX Runtime"
            label = {"cuda": "GPU - CUDA", "tensorrt": "GPU - TensorRT", "directml": "GPU - DirectML",
                     "coreml": "CoreML", "auto": "Auto"}.get(device, device)
            info["device_fallback"] = f"{label} requested -> running on {info['active_device']} ({reason})"
            info["device_fallback_detail"] = "; ".join(steps)
        else:
            info["device_fallback"] = info["device_fallback_detail"] = ""
        return session, active
    raise RuntimeError("the model could not run on any provider: " + "; ".join(steps)
                       + (f" (last error: {last_exc})" if last_exc else ""))


def _high_res_timer(enable: bool) -> bool:
    """Windows' default ~15.6 ms timer granularity would stretch every
    pacing sleep; request 1 ms for this process (no-op elsewhere)."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        (ctypes.windll.winmm.timeBeginPeriod if enable else ctypes.windll.winmm.timeEndPeriod)(1)
        return True
    except Exception:
        return False


class _Stats:
    """Local, lock-free statistics for one model process."""

    def __init__(self, np):
        self.np = np
        self.hist = np.zeros(_HIST_BINS, dtype=np.int64)
        self.overflow = 0
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min = math.inf
        self.max = 0.0
        self.window_start = None     # perf_counter at start of measurement window
        self.last_end = None
        self.instrumentation = 0.0   # seconds spent publishing, excluded from FPS
        self.idle = 0.0              # seconds with no work requested (rate 0), excluded from FPS
        self.recent = deque()
        self.percentiles = {}

    def add(self, start: float, end: float) -> None:
        latency = end - start
        if self.window_start is None:
            self.window_start = start
        self.last_end = end
        self.count += 1
        delta = latency - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (latency - self.mean)
        if latency < self.min:
            self.min = latency
        if latency > self.max:
            self.max = latency
        index = int(latency * 1_000_000)
        if index < _HIST_BINS:
            self.hist[index] += 1
        else:
            self.overflow += 1
        recent = self.recent
        recent.append(end)
        cutoff = end - _FPS_WINDOW
        while recent[0] < cutoff:
            recent.popleft()

    def fps(self) -> float | None:
        if not self.count or self.window_start is None:
            return None
        busy = (self.last_end - self.window_start) - self.instrumentation - self.idle
        return self.count / busy if busy > 0 else None

    def instant_fps(self, now: float) -> float:
        recent = [x for x in self.recent if x >= now - _FPS_WINDOW]
        if len(recent) < 2:
            return 0.0
        return (len(recent) - 1) / (recent[-1] - recent[0])

    def refresh_percentiles(self) -> None:
        if not self.count:
            return
        cumulative = self.np.cumsum(self.hist)
        result = {}
        for label, q in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)):
            rank = max(1, math.ceil(q * self.count))
            index = int(self.np.searchsorted(cumulative, rank))
            if index >= _HIST_BINS:          # falls in the >1 s overflow bucket
                value = self.max
            else:
                value = min(self.max, max(self.min, (index + 0.5) / 1_000_000))
            result[label] = value * 1000.0
        self.percentiles = result


def _model_process(name_id: str, cfg: dict[str, Any], path: str, out_file: str,
                   session_t0: float, stop_event, rate_value=None) -> None:
    """Load and repeatedly run one model in its own OS process."""
    pc = time.perf_counter
    local_stop = threading.Event()
    # multiprocessing.Event.is_set() takes a semaphore every call; a local
    # threading.Event is a plain flag read, so the per-inference stop check
    # costs effectively nothing. A watcher thread bridges the two by POLLING
    # is_set(): it must never block in stop_event.wait(), because a process
    # that exits while registered as a waiter (e.g. a model that failed to
    # load) would leave the parent deadlocked forever inside stop_event.set().
    def _watch_stop():
        while not stop_event.is_set():
            if local_stop.wait(0.05):
                return
        local_stop.set()
    watcher = threading.Thread(target=_watch_stop, daemon=True)
    watcher.start()
    high_res = _high_res_timer(True)

    requested = normalize_device(cfg.get("device", DEFAULT_DEVICE))
    info: dict[str, Any] = {
        "name": name_id, "path": path, "state": "loading", "error": "",
        "provider": "", "requested_device": requested, "device_id": int(cfg.get("device_id", 0) or 0),
        "io_binding": "",
        "warmup_iterations": int(cfg.get("warmup_iterations", DEFAULT_WARMUP)),
        "input_shape": [], "input_layout": "", "input_width": 0, "input_height": 0,
        "load_seconds": None, "cold_start_latency_ms": None, "warmup_inferences": 0,
        "inferences": 0, "failures": 0, "target_rate_fps": None,
    }
    history: list[dict[str, float]] = []
    stats = None

    def snapshot(final: bool = False) -> dict[str, Any]:
        data = dict(info)
        if stats is not None:
            now = pc()
            count = stats.count
            data["inferences"] = count
            data["fps"] = stats.fps()
            data["instant_fps"] = stats.instant_fps(now) if not final else (history[-1]["fps"] if history else stats.instant_fps(now))
            data["latency_mean_ms"] = stats.mean * 1000 if count else None
            data["latency_min_ms"] = stats.min * 1000 if count else None
            data["latency_max_ms"] = stats.max * 1000 if count else None
            data["latency_stdev_ms"] = math.sqrt(stats.m2 / (count - 1)) * 1000 if count > 1 else None
            for label, value in stats.percentiles.items():
                data[f"latency_{label}_ms"] = value
            # Upper bound the hardware could sustain at the measured latency,
            # i.e. with no pacing at all.
            data["compute_fps"] = 1.0 / stats.mean if count and stats.mean > 0 else None
            data["measurement_seconds"] = (stats.last_end - stats.window_start) if count else 0.0
            data["instrumentation_seconds"] = stats.instrumentation
            data["latency_overflow_count"] = stats.overflow
        else:
            data.update({"fps": None, "instant_fps": 0.0, "latency_mean_ms": None,
                         "latency_min_ms": None, "latency_max_ms": None})
        data["updated_at"] = time.time()
        if final:
            data["fps_history"] = history
        return data

    def publish(final: bool = False) -> None:
        publish_json(out_file, snapshot(final))

    try:
        publish()
        import numpy as np
        import onnxruntime as ort

        device_id = int(cfg.get("device_id", 0) or 0)
        hint = int(cfg.get("_cpu_threads_hint", 0) or 0)

        def base_options():
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            for key in ("intra_op_num_threads", "inter_op_num_threads"):
                value = int(cfg.get(key, 0) or 0)
                if value > 0:
                    setattr(options, key, value)
            return options

        load_started = pc()
        session, active = _create_session(ort, requested, cfg, path, base_options, info)
        if active == "CPUExecutionProvider" and hint and not int(cfg.get("intra_op_num_threads", 0) or 0):
            # Shared-process mode on the CPU: each instance's ONNX Runtime
            # would otherwise size its thread pool to ALL physical cores, so
            # N instances oversubscribe the CPU N times over. Re-create the
            # CPU session with this instance's share of the cores.
            options = base_options()
            options.intra_op_num_threads = hint
            session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
            info["intra_op_threads"] = hint
        info["load_seconds"] = pc() - load_started
        if info.get("device_fallback"):
            print(f"[ai] {name_id}: {info['device_fallback']}", flush=True)

        model_input = session.get_inputs()[0]
        input_name = model_input.name
        input_shape = list(model_input.shape)
        tensor, layout, width, height = _input_tensor(np, input_shape, model_input.type)
        output_names = [o.name for o in session.get_outputs()]

        run = None
        if active in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
            try:
                device_input = ort.OrtValue.ortvalue_from_numpy(tensor, "cuda", device_id)
                binding = session.io_binding()
                binding.bind_ortvalue_input(input_name, device_input)
                for name in output_names:
                    binding.bind_output(name, "cuda", device_id)
                run_with_binding = session.run_with_iobinding
                sync = binding.synchronize_outputs

                def run():
                    run_with_binding(binding)
                    sync()
                run()  # validate the binding once before relying on it
                info["io_binding"] = "device (input/outputs resident on GPU)"
            except Exception as exc:
                run = None
                info["io_binding"] = f"host (device binding unavailable: {exc})"
        if run is None:
            feeds = {input_name: tensor}
            session_run = session.run

            def run():
                session_run(output_names, feeds)
            info["io_binding"] = info["io_binding"] or "host"

        info.update({"state": "warming_up", "provider": active, "input_shape": input_shape,
                     "input_layout": layout, "input_width": width, "input_height": height})
        stats = _Stats(np)
        publish()
    except Exception as exc:
        info["state"] = "failed"
        info["error"] = str(exc)
        publish(final=True)
        local_stop.set()
        if high_res:
            _high_res_timer(False)
        return

    # Pacing: when the parent assigns this instance a share of the model's
    # target throughput (rate_value > 0), inferences are started on an
    # absolute schedule of 1/rate seconds. The schedule is absolute so sleep
    # jitter averages out exactly; if the instance falls more than one period
    # behind it does NOT burst to catch up -- it just runs back-to-back, which
    # is exactly the "this instance can't keep up" signal the scaler reads.
    # rate_value is a lock-free shared double, so reading it every cycle
    # costs ~100 ns. With no rate (no target) inferences run back-to-back.
    read_rate = (lambda: rate_value.value) if rate_value is not None else (lambda: 0.0)
    next_due = None
    current_period = None
    warmup = max(0, int(cfg.get("warmup_iterations", DEFAULT_WARMUP)))
    warmed = 0
    measuring = warmup == 0
    if measuring:
        info["state"] = "running"
    next_publish = pc() + _PUBLISH_INTERVAL
    next_percentiles = pc() + _PERCENTILE_INTERVAL
    is_stopped = local_stop.is_set

    while not is_stopped():
        rate = read_rate()
        if rate > 0:
            period = 1.0 / rate
            now = pc()
            if next_due is None or period != current_period:
                current_period = period
                info["target_rate_fps"] = rate
                next_due = now
            delay = next_due - now
            if delay > 0:
                if local_stop.wait(delay):
                    break
            elif delay < -period:
                next_due = now          # behind schedule: no catch-up burst
            next_due += period
        started = pc()
        try:
            run()
            ended = pc()
        except Exception as exc:
            info["failures"] += 1
            info["error"] = str(exc)
            if local_stop.wait(0.1):
                break
            continue

        if not measuring:
            warmed += 1
            info["warmup_inferences"] = warmed
            if warmed == 1:
                info["cold_start_latency_ms"] = (ended - started) * 1000
            if warmed >= warmup:
                measuring = True
                info["state"] = "running"
                stats.window_start = ended   # window opens when warm-up ends
            continue

        stats.add(started, ended)

        if ended >= next_publish:
            bookkeeping = pc()
            if bookkeeping >= next_percentiles:
                stats.refresh_percentiles()
                next_percentiles = bookkeeping + _PERCENTILE_INTERVAL
            history.append({"elapsed_seconds": time.time() - session_t0,
                            "fps": stats.instant_fps(bookkeeping)})
            publish()
            next_publish = bookkeeping + _PUBLISH_INTERVAL
            stats.instrumentation += pc() - bookkeeping

    stats.refresh_percentiles()   # exact over every measured inference
    info["state"] = "stopped"
    publish(final=True)
    if high_res:
        _high_res_timer(False)


# --------------------------------------------------------------------------- #
# Parent-side: instances, per-model scaling, workload
# --------------------------------------------------------------------------- #
SCALER_VERSION = "7 (min gain always on -> saturated; lowering instance stopped; device fallback to CPU)"
DEFAULT_TARGET_FPS_PER_STREAM = 15.0
DEFAULT_AUTOSCALE = {
    "enabled": True,
    "max_instances": 16,         # per model
    "settle_seconds": 8.0,       # steady time after an instance comes up before judging
    "tolerance_percent": 2.0,    # total within this much of target counts as met
    "min_gain_percent": 5.0,     # every added instance must raise the total by this much, else
                                 # scaling ends (always enforced, must be > 0)
}


class _SlotValue:
    """`.value` view of one element of a shared lock-free RawArray, so a
    worker thread reads its pacing rate exactly like a process reads its
    own RawValue."""
    __slots__ = ("arr", "i")

    def __init__(self, arr, i):
        self.arr, self.i = arr, i

    @property
    def value(self):
        return self.arr[self.i]

    @value.setter
    def value(self, v):
        self.arr[self.i] = v


class _SlotFlag:
    """`.is_set()` for one worker thread: its own stop slot, or the host's."""
    __slots__ = ("arr", "i", "host")

    def __init__(self, arr, i, host):
        self.arr, self.i, self.host = arr, i, host

    def is_set(self):
        return self.arr[self.i] != 0 or self.host.is_set()


def _model_host(model_name: str, cfg: dict[str, Any], path: str, session_t0: float,
                host_stop, commands, rates, stops) -> None:
    """ONE OS process hosting every instance of one model as a thread.

    Why: separate processes each get their own GPU context, and on Windows
    the GPU time-slices between contexts -- kernels from different
    processes never overlap (NVIDIA MPS, which would allow it, is
    Linux-only) -- so an added process mostly adds context switching and
    another copy of the CUDA/cuDNN state in VRAM. Threads in one process
    share one context; each instance keeps its own ONNX Runtime session and
    therefore its own CUDA stream, so their GPU work can overlap. ONNX
    Runtime releases Python's GIL while a model runs, so threads do not
    serialize each other. Each instance still measures and publishes
    exactly as before (same file, same fields)."""
    import queue as _queue
    local = threading.Event()
    threads = []
    while not host_stop.is_set():          # poll only -- never block in Event.wait()
        try:
            cmd = commands.get(timeout=0.2)
        except _queue.Empty:
            continue
        except (EOFError, OSError):
            break
        if cmd and cmd[0] == "add":
            _, idx, name_id, out_file, hint = cmd
            c = dict(cfg, _cpu_threads_hint=hint)
            t = threading.Thread(target=_model_process, name=name_id, daemon=True,
                                 args=(name_id, c, path, out_file, session_t0,
                                       _SlotFlag(stops, idx, local), _SlotValue(rates, idx)))
            t.start()
            threads.append(t)
    local.set()
    for t in threads:
        t.join(15)


class ModelHost:
    """Parent-side handle of one model's shared host process."""

    def __init__(self, model_name: str, cfg: dict[str, Any], path: str, session_t0: float, capacity: int):
        self.model_name, self.cfg, self.path, self.session_t0 = model_name, cfg, path, session_t0
        self.capacity = capacity
        self.rates = multiprocessing.RawArray("d", capacity)
        self.stops = multiprocessing.RawArray("b", capacity)
        self.commands = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()
        self.process = None

    def add(self, inst: "ModelInstance", hint: int) -> None:
        if self.process is None:
            self.process = multiprocessing.Process(
                target=_model_host, name=f"ai-host-{self.model_name}", daemon=True,
                args=(self.model_name, self.cfg, self.path, self.session_t0, self.stop_event,
                      self.commands, self.rates, self.stops))
            self.process.start()
        self.commands.put(("add", inst.index, inst.name_id, str(inst.out_file), hint))

    def stop(self) -> None:
        for i in range(self.capacity):
            self.stops[i] = 1
        self.stop_event.set()

    def join(self, timeout: float) -> None:
        if self.process is not None:
            self.process.join(timeout)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2)
        try:
            self.commands.cancel_join_thread()
            self.commands.close()
        except Exception:
            pass


class ModelInstance:
    """One copy of a model. Process mode: its own OS process. Shared mode
    (see SharedModelInstance): a thread in the model's host process."""
    worker = "process"

    def __init__(self, model_name: str, index: int, cfg: dict[str, Any], path: str, out_dir: Path, session_t0: float):
        self.model_name = model_name
        self.index = index
        self.name_id = f"{model_name}#{index}"
        self.cfg = cfg
        self.path = path
        self.out_file = Path(out_dir) / "ai" / f"{slug(model_name)}-i{index}.json"
        self.session_t0 = session_t0
        self.rate = multiprocessing.RawValue("d", 0.0)   # lock-free share of the target
        self.stop_event = None
        self.process = None
        self.started_at = None
        self.running_since = None
        self.retired = False
        self.retired_reason = ""
        self.saturation_note = ""
        self._last_good = None

    def start(self) -> None:
        self.out_file.parent.mkdir(parents=True, exist_ok=True)
        self.stop_event = multiprocessing.Event()
        self.process = multiprocessing.Process(
            target=_model_process,
            args=(self.name_id, self.cfg, self.path, str(self.out_file), self.session_t0, self.stop_event, self.rate),
            name=f"ai-{self.name_id}", daemon=True)
        self.process.start()
        self.started_at = time.time()

    def stop(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()

    def join(self, timeout: float) -> None:
        if self.process is None:
            return
        self.process.join(timeout)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(2)

    def snapshot(self) -> dict[str, Any]:
        try:
            data = json.loads(self.out_file.read_text(encoding="utf-8"))
            self._last_good = data
            data["worker"] = self.worker
        except Exception:
            # On Windows a read can collide with the child replacing the file
            # (PermissionError / half-written JSON). Reuse the last good
            # snapshot: previously this returned a zero-FPS "starting" stub,
            # which dropped the instance from that second's total and made
            # averages -- and therefore gain decisions -- jump randomly.
            data = dict(self._last_good) if self._last_good else {
                "name": self.name_id, "state": "starting", "error": "", "inferences": 0, "failures": 0,
                "fps": None, "instant_fps": 0.0, "latency_mean_ms": None}
        exit_code = self.process.exitcode if self.process is not None else None
        if exit_code not in (None, 0) and data.get("state") not in ("stopped", "failed"):
            data["state"] = "failed"
            data["error"] = data.get("error") or f"model process exited with code {exit_code}"
        if data.get("state") == "running" and self.running_since is None:
            self.running_since = time.time()
        data["instance"] = self.index
        if self.saturation_note:
            data["saturation_note"] = self.saturation_note
        data["assigned_fps"] = self.rate.value or None
        if self.retired:
            data["retired"] = True
            data["retired_reason"] = self.retired_reason
        return data


class SharedModelInstance(ModelInstance):
    """One copy of a model running as a thread in the model's host process."""
    worker = "thread (shared process)"

    def __init__(self, model_name, index, cfg, path, out_dir, session_t0, host: ModelHost, cpu_hint: int):
        super().__init__(model_name, index, cfg, path, out_dir, session_t0)
        if index >= host.capacity:
            raise RuntimeError(f"shared host capacity {host.capacity} exceeded")
        self.host = host
        self.cpu_hint = cpu_hint
        self.rate = _SlotValue(host.rates, index)

    def start(self) -> None:
        self.out_file.parent.mkdir(parents=True, exist_ok=True)
        self.host.add(self, self.cpu_hint)
        self.process = self.host.process      # exit-code checks see the host
        self.started_at = time.time()

    def stop(self) -> None:
        self.host.stops[self.index] = 1

    def join(self, timeout: float) -> None:
        pass                                   # the host process is joined by its group


class ModelGroup:
    """All instances of one model plus the scaler that decides how many are
    needed to reach target = target_fps_per_stream x enabled input streams.

    Search (one instance at a time, so the answer is the MINIMUM count):
      1. start with ``instances`` copies (default 1), each paced to target/N;
      2. once every instance is running (warm-up done) plus 5 s, measure the
         total over a settle_seconds window. The total is computed from each
         instance's cumulative inference counter and its own timestamps
         (exact completed-inferences-per-second), never from rolling or
         single readings;
      3. minimum gain (always on, applies to EVERY added instance): if the
         newest instance raised the measured total by less than
         min_gain_percent over the total before it was added, no more
         instances are ever added and the model is "saturated" (or
         "target_met" if the total meets the target). If the newest instance
         actually LOWERED the total, it is stopped and the target is re-split
         over the remaining instances; if it helped a little (0% to the
         minimum), it keeps running;
         otherwise, target met -> done (keeps monitoring: a sustained
                           shortfall adds another instance, which must again
                           pass the minimum gain);
         short          -> add one instance, re-split the target, go to 2;
         max_instances reached -> stop, report the shortfall and best total;
         a new instance fails to start (e.g. out of VRAM) -- it has already
                           exited on its own; further scaling stops.
    """

    def __init__(self, name: str, cfg: dict[str, Any], ai_cfg: dict[str, Any], base: Path, out_dir: Path, session_t0: float):
        self.name = name
        self.cfg = cfg
        path = Path(cfg["path"])
        self.path = str(path if path.is_absolute() else (base / path).resolve())
        self.out_dir = Path(out_dir)
        self.session_t0 = session_t0
        self.per_stream = float(cfg.get("target_fps_per_stream", ai_cfg.get("target_fps_per_stream", DEFAULT_TARGET_FPS_PER_STREAM)) or 0)
        scale = dict(DEFAULT_AUTOSCALE)
        scale.update(ai_cfg.get("autoscale", {}) or {})
        scale.update(cfg.get("autoscale", {}) or {})
        self.scale = scale
        self.initial = max(1, int(cfg.get("instances", 1) or 1))
        # "shared" (default): all instances of this model are threads in one
        # host process (one GPU context). "process": one OS process each.
        self.worker_mode = str(cfg.get("worker_mode", ai_cfg.get("worker_mode", "shared"))).lower()
        self.host = None
        try:
            import psutil
            phys = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
        except Exception:
            phys = os.cpu_count() or 1
        # CPU threads per instance in shared mode: split the physical cores
        # across the most instances this model may run.
        self.cpu_hint = max(1, phys // max(1, min(int(scale.get("max_instances", 16)), phys)))
        self.instances: list[ModelInstance] = []
        self.retired: list[ModelInstance] = []
        self.stream_count = 0
        self.target = 0.0
        self.state = "idle"
        self.reason = ""
        self.required_instances = None
        self.last_total = None          # total FPS before the most recent scale-up
        self.settle_from = None
        self.below_since = None
        self.events: list[dict[str, Any]] = []
        self.step_totals: list[dict[str, float]] = []   # settled total FPS at each instance count
        self.records = deque()                            # (time, {instance: (inferences, child timestamp)})
        self.eval_start = None                            # start of the current measurement window
        self.locked = False                               # min-gain rule ended scaling for good
        self.target_reached = False                       # target met at the moment scaling was locked
        self.measured_rate = None                         # counter-based total FPS over the last 5 s
        self.history: list[dict[str, float]] = []
        self.last_snapshots: list[dict[str, Any]] = []

    # --- helpers -----------------------------------------------------------
    def _log(self, action: str, total: float | None, reason: str) -> None:
        event = {"elapsed_seconds": round(time.time() - self.session_t0, 2), "action": action,
                 "instances": len(self.instances), "total_fps": total, "target_fps": self.target, "reason": reason}
        self.events.append(event)
        print(f"[ai] {self.name}: {action} -> {len(self.instances)} instance(s)"
              + (f", total {total:.2f}/{self.target:.2f} FPS" if total is not None else "") + f" ({reason})", flush=True)

    def _rebalance(self) -> None:
        share = self.target / len(self.instances) if self.instances and self.target > 0 else 0.0
        for inst in self.instances:
            inst.rate.value = share

    def _add(self, reason: str) -> None:
        if self.locked:   # hard guard: nothing may add an instance after a min-gain failure
            return
        index = len(self.instances) + len(self.retired) + 1
        if self.worker_mode == "shared":
            if self.host is None:
                self.host = ModelHost(self.name, self.cfg, self.path, self.session_t0,
                                      capacity=int(self.scale.get("max_instances", 16)) * 2 + self.initial + 8)
            inst = SharedModelInstance(self.name, index, self.cfg, self.path, self.out_dir, self.session_t0,
                                       self.host, self.cpu_hint)
        else:
            inst = ModelInstance(self.name, index, self.cfg, self.path, self.out_dir, self.session_t0)
        self.instances.append(inst)
        self._rebalance()
        inst.start()
        self.settle_from = None
        self.eval_start = None
        self._log("add_instance", self.last_total, reason)

    def _retire(self, inst: ModelInstance, reason: str) -> None:
        inst.retired, inst.retired_reason = True, reason
        inst.stop()
        self.instances.remove(inst)
        self.retired.append(inst)
        self._rebalance()

    @property
    def min_gain(self) -> float:
        """Always enforced; a missing, zero or invalid value falls back to the default."""
        try:
            value = float(self.scale.get("min_gain_percent", DEFAULT_AUTOSCALE["min_gain_percent"]))
        except (TypeError, ValueError):
            value = 0.0
        return (value if value > 0 else DEFAULT_AUTOSCALE["min_gain_percent"]) / 100.0

    @property
    def autoscale(self) -> bool:
        return bool(self.scale.get("enabled", True)) and self.target > 0

    # --- lifecycle ---------------------------------------------------------
    def start(self, stream_count: int) -> None:
        self.stream_count = stream_count
        self.target = self.per_stream * stream_count
        self.state = "scaling" if self.autoscale else "fixed"
        if self.target <= 0:
            self.reason = "no FPS target (target_fps_per_stream is 0 or no AI-enabled streams): instances run unpaced"
        for _ in range(self.initial):
            self._add("initial instance")

    # --- measurement ---------------------------------------------------------
    def _record(self, now: float, snaps: list[dict[str, Any]]) -> None:
        """Store each instance's cumulative inference counter and the child's
        own timestamp for it. Rates are computed from counter DIFFERENCES,
        not from the children's rolling 'instant_fps' values: those are only
        refreshed when an inference completes, so a process starved by a
        newly added one kept reporting its old, higher rate, and the sum of
        such stale values made additions look like real gains."""
        self.records.append((now, {inst.index: (int(s.get("inferences") or 0), float(s.get("updated_at") or 0))
                                   for inst, s in zip(self.instances, snaps) if s.get("state") == "running"}))
        horizon = now - max(60.0, 3 * float(self.scale["settle_seconds"]))
        while len(self.records) > 2 and self.records[0][0] < horizon:
            self.records.popleft()

    def _rate_since(self, start: float) -> tuple[float, float]:
        """Total inferences/s across instances from the first record at or
        after `start` to the latest record, and the span covered."""
        first = next((r for r in self.records if r[0] >= start), None)
        if first is None or not self.records:
            return 0.0, 0.0
        last = self.records[-1]
        total = 0.0
        for idx, (count_b, time_b) in last[1].items():
            if idx not in first[1]:
                continue
            count_a, time_a = first[1][idx]
            if time_b > time_a:
                total += (count_b - count_a) / (time_b - time_a)
        return total, last[0] - first[0]

    def tick(self) -> dict[str, Any]:
        snaps = [inst.snapshot() for inst in self.instances]
        self.last_snapshots = snaps
        now = time.time()
        live_total = sum(float(s.get("instant_fps") or 0) for s in snaps if s.get("state") == "running")
        self.history.append({"elapsed_seconds": round(now - self.session_t0, 2), "fps": live_total})
        self._record(now, snaps)
        # Exact throughput over the last 5 s from inference counters; this is
        # what the persistence workload follows (one save per inference).
        rate, span = self._rate_since(now - _FPS_WINDOW)
        self.measured_rate = rate if span >= 1.0 else None

        if not self.autoscale or self.state in ("saturated", "instance_failed", "max_instances", "failed") or self.locked:
            return self.snapshot()

        failed = [(i, s) for i, s in zip(self.instances, snaps) if s.get("state") == "failed"]
        if failed:
            inst, snap = failed[-1]
            if len(self.instances) == 1:
                self.state, self.reason = "failed", snap.get("error", "model failed")
                self._log("failed", None, self.reason)
                return self.snapshot()
            # The process already exited by itself; it is only moved out of
            # the active list so the target is re-split over working ones.
            self._retire(inst, f"failed to start: {snap.get('error', '')}")
            self.state = "instance_failed"
            self.reason = (f"instance #{inst.index} could not start ({snap.get('error', '')}); "
                           f"running with {len(self.instances)} instance(s)")
            self._log("instance_failed", self.last_total, self.reason)
            return self.snapshot()

        if not all(i.running_since is not None for i in self.instances):
            self.settle_from = None
            self.eval_start = None
            return self.snapshot()
        settle = max(3.0, float(self.scale["settle_seconds"]))
        tol = float(self.scale["tolerance_percent"]) / 100.0
        if self.settle_from is None:
            # Give the newest instance a few seconds of steady running (and
            # the others time to adapt to the new, smaller pacing share)
            # before the measurement window opens.
            self.settle_from = max(i.running_since for i in self.instances) + _FPS_WINDOW

        if self.state == "target_met":
            rate, span = self._rate_since(now - settle)
            if span >= settle - 1.01 and rate < self.target * (1 - tol):
                self.state = "scaling"
                self.required_instances = None
                self._log("target_lost", rate, f"{settle:g} s average {rate:.2f} FPS fell below the target")
                return self._scale_up_or_stop(rate)
            return self.snapshot()

        if now < self.settle_from:
            return self.snapshot()
        if self.eval_start is None:
            self.eval_start = now
            return self.snapshot()
        rate, span = self._rate_since(self.eval_start)
        if span < settle:
            return self.snapshot()
        self.eval_start = None
        avg = rate
        self.step_totals.append({"instances": len(self.instances), "total_fps": avg})
        met = avg >= self.target * (1 - tol)
        gain = self.min_gain

        if self.last_total is not None:
            # The minimum-gain rule applies to EVERY added instance, including
            # one that happens to reach the target: an instance that adds
            # less than min_gain_percent ends scaling for good.
            change = 100 * (avg - self.last_total) / self.last_total if self.last_total else 0.0
            needed = self.last_total * (1 + gain)
            print(f"[ai] {self.name}: {len(self.instances)} instance(s) measured {avg:.2f} FPS over {span:.0f} s "
                  f"({change:+.1f}% vs {self.last_total:.2f}; min gain {gain * 100:g}% needs >= {needed:.2f})", flush=True)
            if avg < needed:
                # The newest instance did not add the minimum gain: the
                # hardware is saturated for this model. The model is marked
                # "saturated" (or "target_met" if the target is nevertheless
                # met), no instance is ever added again (self.locked is also
                # checked right before any add), and nothing is stopped --
                # the instance is healthy, it just didn't add throughput.
                newest = self.instances[-1]
                self.locked = True
                best = max(x["total_fps"] for x in self.step_totals)
                if avg < self.last_total:
                    # Adding the instance made the model SLOWER (contention
                    # outweighed the extra worker): stop it, go back to the
                    # previous count -- the target is re-split over the
                    # remaining instances -- and mark the model saturated.
                    self._retire(newest, f"stopped: adding it lowered the total {self.last_total:.2f} -> {avg:.2f} FPS ({change:+.1f}%)")
                    met = self.last_total >= self.target * (1 - tol)
                    kept = (f"Instance #{newest.index} was stopped; running with {len(self.instances)} instance(s) "
                            f"(~{self.last_total:.2f} FPS). ")
                else:
                    # It helped a little, just less than the minimum: keep it.
                    newest.saturation_note = f"added {change:+.1f}% (below the {gain * 100:g}% minimum gain) - model saturated"
                    kept = f"{len(self.instances)} instance(s) keep running. "
                self.state = "target_met" if met else "saturated"
                self.target_reached = met
                self.required_instances = len(self.instances) if met else None
                self.reason = (f"Saturated: adding instance #{newest.index} changed the total {self.last_total:.2f} -> "
                               f"{avg:.2f} FPS ({change:+.1f}%), below the {gain * 100:g}% minimum gain. No more instances will be added. "
                               + kept
                               + (f"Target met." if met else
                                  f"Best total {best:.2f} FPS = {100 * best / self.target:.0f}% of the {self.target:.2f} FPS target."))
                self._log("saturated", avg, self.reason)
                return self.snapshot()

        if met:
            self.state = "target_met"
            self.required_instances = len(self.instances)
            self.reason = f"{len(self.instances)} instance(s) sustain {avg:.2f} FPS for a {self.target:.2f} FPS target"
            self._log("target_met", avg, self.reason)
            return self.snapshot()
        return self._scale_up_or_stop(avg)

    def _scale_up_or_stop(self, avg: float) -> dict[str, Any]:
        if len(self.instances) >= int(self.scale["max_instances"]):
            self.state = "max_instances"
            best = max(self.step_totals or [{"instances": len(self.instances), "total_fps": avg}], key=lambda x: x["total_fps"])
            self.reason = (f"reached the max of {self.scale['max_instances']} instance(s) at {avg:.2f} of {self.target:.2f} FPS "
                           f"({100 * avg / self.target:.0f}% of target); best total {best['total_fps']:.2f} FPS "
                           f"with {best['instances']} instance(s)")
            self._log("max_instances", avg, self.reason)
            return self.snapshot()
        self.last_total = avg
        self._add(f"{avg:.2f} FPS average below target {self.target:.2f} FPS")
        return self.snapshot()

    def stop(self) -> None:
        for inst in self.instances + self.retired:
            inst.stop()
        if self.host is not None:
            self.host.stop()

    def join(self, timeout: float) -> None:
        for inst in self.instances + self.retired:
            inst.join(timeout)
        if self.host is not None:
            self.host.join(timeout)

    def snapshot(self, final: bool = False) -> dict[str, Any]:
        snaps = [inst.snapshot() for inst in self.instances] if final or not self.last_snapshots else self.last_snapshots
        retired = [inst.snapshot() for inst in self.retired]
        running = [s for s in snaps if s.get("state") in ("running", "stopped")]
        total_instant = sum(float(s.get("instant_fps") or 0) for s in snaps if s.get("state") == "running")
        total_inf = sum(int(s.get("inferences") or 0) for s in running)
        weighted = [(s["latency_mean_ms"], s.get("inferences") or 0) for s in running if s.get("latency_mean_ms") is not None]
        latency_mean = (sum(v * n for v, n in weighted) / sum(n for _, n in weighted)) if weighted and sum(n for _, n in weighted) else None
        worst = lambda k: max((s[k] for s in running if s.get(k) is not None), default=None)
        # Average total since the instance count last changed (steady state).
        change_at = self.events[-1]["elapsed_seconds"] if self.events else 0
        steady = [h["fps"] for h in self.history if h["elapsed_seconds"] >= change_at + _FPS_WINDOW + 1]
        if not steady:
            steady = [h["fps"] for h in self.history[-10:] if h["fps"] > 0]
        first = snaps[0] if snaps else {}
        data = {
            "name": self.name, "path": self.path, "state": self.state, "scaling_reason": self.reason,
            "error": "; ".join(s.get("error") for s in snaps if s.get("error")),
            "provider": first.get("provider", ""), "requested_device": normalize_device(self.cfg.get("device", DEFAULT_DEVICE)),
            "active_device": first.get("active_device", ""), "device_fallback": first.get("device_fallback", ""),
            "device_fallback_detail": first.get("device_fallback_detail", ""),
            "target_fps_per_stream": self.per_stream, "stream_count": self.stream_count, "target_fps": self.target,
            "autoscale": self.autoscale, "instance_count": len(self.instances), "required_instances": self.required_instances,
            "instant_fps": total_instant,
            "measured_fps": self.measured_rate,
            "fps": (sum(steady) / len(steady)) if steady else None,
            "target_met": (self.target > 0 and (self.state == "target_met" or self.target_reached)),
            "scaler_version": SCALER_VERSION,
            "worker_mode": self.worker_mode,
            "inferences": total_inf, "failures": sum(int(s.get("failures") or 0) for s in snaps),
            "latency_mean_ms": latency_mean, "latency_p95_ms": worst("latency_p95_ms"),
            "latency_p99_ms": worst("latency_p99_ms"), "latency_max_ms": worst("latency_max_ms"),
            "compute_fps": sum(float(s.get("compute_fps") or 0) for s in running) or None,
            "input_shape": first.get("input_shape"), "io_binding": first.get("io_binding"),
            "instances": snaps, "retired_instances": retired, "scaling_events": self.events,
            "scaling_steps": self.step_totals,
        }
        if final:
            data["fps_history"] = self.history
        else:
            for s in data["instances"] + data["retired_instances"]:
                s.pop("fps_history", None)
        return data


class ModelWorkload:
    def __init__(self, ai_cfg: dict[str, Any], base: Path, out_dir: Path, session_t0: float | None = None):
        t0 = session_t0 if session_t0 is not None else time.time()
        ai_cfg = ai_cfg or {}
        self.groups = [
            ModelGroup(name, cfg, ai_cfg, base, Path(out_dir), t0)
            for name, cfg in ai_cfg.get("models", {}).items()
            if isinstance(cfg, dict) and cfg.get("enabled", False)
        ]

    @property
    def workers(self):   # kept for callers that only test "any models?"
        return self.groups

    def start(self, stream_count: int = 1) -> None:
        for group in self.groups:
            group.start(stream_count)

    def tick(self) -> list[dict[str, Any]]:
        return [group.tick() for group in self.groups]

    def stop(self) -> None:
        for group in self.groups:
            group.stop()
        for group in self.groups:
            group.join(15)

    def close(self) -> None:
        pass

    def snapshot(self, final: bool = False) -> list[dict[str, Any]]:
        return [group.snapshot(final=final) for group in self.groups]
