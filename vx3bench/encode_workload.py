"""Encode test: N independent live video encode streams, 1080p30 or 4K30.

Each stream is its own OS process. It feeds raw YUV 4:2:0 frames with
moving content, at exactly 30 FPS on an absolute schedule, into its own
FFmpeg encoder and drains the encoded H.264 bitstream -- the same shape of
work as encoding one live camera.

Encoder choice (per stream):
  1. a hardware (GPU) H.264 encoder, if one works on this machine -- probed
     once at start in this order: NVIDIA NVENC, Intel Quick Sync, AMD AMF,
     Apple VideoToolbox, Linux VA-API, ARM V4L2 M2M, Windows Media
     Foundation (hardware MFT, incl. Windows on ARM);
  2. otherwise the CPU encoder (libx264, preset veryfast, zerolatency).
  If a stream's GPU encoder fails to open (e.g. the NVIDIA consumer-GPU
  limit on concurrent NVENC sessions), THAT stream falls back to the CPU and
  the report says why.

Measurement:
  * encoded FPS comes from the encoder's own frame counter (FFmpeg
    -progress, every 0.5 s), differenced over time: exact frames that left
    the encoder, not frames we tried to send;
  * the first 2 s after the first encoded frame are warm-up and excluded;
  * a live source cannot wait: if the encoder falls behind, the frame slot
    is missed (counted as dropped) rather than sent late in a burst;
  * keep-up = encoded frames / frames due at 30 FPS over the measurement
    window; bitrate comes from the actual encoded bytes;
  * frame hand-off time (write into the encoder's input pipe) is timed
    with perf_counter; when it grows, the encoder is back-pressuring.

FFmpeg is located via the VX3_FFMPEG environment variable, the
imageio-ffmpeg package (pip install imageio-ffmpeg, bundles a build with
NVENC/QSV/AMF and libx264), or ffmpeg on PATH.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from .live_publish import publish_json
from .model_workers import _Stats, _high_res_timer

PRESETS = {
    "1080p30": {"width": 1920, "height": 1080, "fps": 30, "bitrate": "8M", "label": "1080p30"},
    "4k30": {"width": 3840, "height": 2160, "fps": 30, "bitrate": "25M", "label": "4K30"},
}
# (ffmpeg encoder, label, encoder options, options before the input, video filter options)
# Tried in this order; each is probed with a real 10-frame encode, so only
# encoders that WORK on this machine (hardware + driver + FFmpeg build) are
# used. The list covers Windows (x64 and ARM), Linux (x64 and ARM) and macOS.
GPU_ENCODERS = [
    ("h264_nvenc", "NVIDIA NVENC", ["-preset", "p4", "-tune", "ll", "-rc", "cbr"], [], []),
    ("h264_qsv", "Intel Quick Sync", ["-preset", "veryfast"], [], []),
    ("h264_amf", "AMD AMF", ["-usage", "lowlatency", "-quality", "speed"], [], []),
    ("h264_videotoolbox", "Apple VideoToolbox", ["-realtime", "1"], [], []),
    ("h264_vaapi", "VA-API (Linux Intel/AMD)", [], ["-vaapi_device", "/dev/dri/renderD128"],
     ["-vf", "format=nv12,hwupload"]),
    ("h264_v4l2m2m", "V4L2 M2M (ARM boards, e.g. Raspberry Pi)", [], [], []),
    ("h264_mf", "Windows Media Foundation (hardware, incl. Windows on ARM)", ["-hw_encoding", "1"], [], []),
]
CPU_ENCODER = ("libx264", "CPU (libx264)", ["-preset", "veryfast", "-tune", "zerolatency"], [], [])
WARMUP_SECONDS = 2.0
_WIN_FLAGS = 0x08000000 if os.name == "nt" else 0   # CREATE_NO_WINDOW


def ffmpeg_candidates() -> list[str]:
    """Every FFmpeg we can find, in preference order (duplicates removed)."""
    found = []
    env = os.environ.get("VX3_FFMPEG")
    if env and Path(env).exists():
        found.append(env)
    on_path = shutil.which("ffmpeg")
    if on_path:
        found.append(on_path)
    try:
        import imageio_ffmpeg
        found.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    out = []
    for f in found:
        if f not in out:
            out.append(f)
    return out


def find_ffmpeg() -> str | None:
    c = ffmpeg_candidates()
    return c[0] if c else None


def _encoder_args(name: str, opts: list[str], bitrate: str, fps: int) -> list[str]:
    rate = ["-b:v", bitrate, "-maxrate", bitrate, "-bufsize", bitrate]
    return ["-c:v", name, *opts, *rate, "-g", str(2 * fps), "-bf", "0"]


def probe_gpu_encoder(ffmpeg: str, width: int, height: int, fps: int) -> tuple[tuple | None, list[str]]:
    """First hardware encoder that can really encode a few frames here."""
    notes = []
    try:
        listed = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True,
                                timeout=20, creationflags=_WIN_FLAGS).stdout
    except Exception as exc:
        return None, [f"could not list encoders: {exc}"]
    for enc in GPU_ENCODERS:
        name = enc[0]
        if name not in listed:
            notes.append(f"{enc[1]}: not in this FFmpeg build")
            continue
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", *enc[3], "-f", "lavfi", "-i",
               f"testsrc2=size={width}x{height}:rate={fps}", "-frames:v", "10", *enc[4],
               *_encoder_args(name, enc[2], "8M", fps), "-f", "null", "-"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, creationflags=_WIN_FLAGS)
            if r.returncode == 0:
                notes.append(f"{enc[1]}: OK")
                return enc, notes
            lines = [l.strip() for l in (r.stderr or "").splitlines() if l.strip()]
            notes.append(f"{enc[1]}: not usable ({' / '.join(lines[:2])[:200] or 'failed'})")
        except Exception as exc:
            notes.append(f"{enc[1]}: {exc}")
    return None, notes


# --------------------------------------------------------------------------- #
# Child process
# --------------------------------------------------------------------------- #
def _frames(np, width: int, height: int, count: int):
    """`count` distinct YUV420p frames with motion (a moving textured band
    over gradients), so the encoder does real motion-estimation work."""
    y0 = (np.add.outer(np.arange(height) * 0.5, np.arange(width) * 0.25) % 256).astype(np.uint8)
    rng = np.random.default_rng(7)
    tex = rng.integers(16, 235, size=(height // 4, width), dtype=np.uint8)
    cw, ch = width // 2, height // 2
    u = np.full((ch, cw), 110, np.uint8)
    v = np.full((ch, cw), 150, np.uint8)
    out = []
    for k in range(count):
        y = y0.copy()
        top = (k * height // (count * 2)) % (height - tex.shape[0])
        y[top:top + tex.shape[0]] = np.roll(tex, k * 24, axis=1)
        out.append(y.tobytes() + np.roll(u, k * 6, axis=1).tobytes() + np.roll(v, k * 6, axis=0).tobytes())
    return out


def _encode_host(names: list[str], spec: dict[str, Any], out_files: list[str], session_t0: float, host_stop) -> None:
    """Shared mode: ONE OS process feeding every encode stream, one thread
    per stream. Each stream still has its own FFmpeg encoder process (one
    encoder per camera, as in production); only the Python feeders are
    consolidated. Writing into an encoder's pipe releases Python's GIL, so
    the feeders don't hold each other up, and they share ONE set of test
    frames instead of each process holding its own copy (~24 MB per 1080p
    stream, ~50 MB per 4K stream) plus its own Python/numpy runtime."""
    import numpy as np
    frames = _frames(np, spec["width"], spec["height"], 8 if spec["width"] <= 1920 else 4)
    local = threading.Event()
    threads = [threading.Thread(target=_encode_stream_process, name=n, daemon=True,
                                args=(n, spec, f, session_t0, local), kwargs={"frames": frames})
               for n, f in zip(names, out_files)]
    for t in threads:
        t.start()
    while not host_stop.is_set():      # poll only -- never block in Event.wait()
        time.sleep(0.1)
    local.set()
    for t in threads:
        t.join(20)


def _encode_stream_process(name: str, spec: dict[str, Any], out_file: str, session_t0: float, stop_event, _rate=None,
                           frames=None) -> None:
    import numpy as np
    pc = time.perf_counter
    local_stop = threading.Event()

    def _watch():
        while not stop_event.is_set():
            if local_stop.wait(0.05):
                return
        local_stop.set()
    threading.Thread(target=_watch, daemon=True).start()
    high_res = _high_res_timer(True)

    width, height, fps = spec["width"], spec["height"], spec["fps"]
    info: dict[str, Any] = {"name": name, "kind": "encode_stream", "state": "starting", "error": "",
                            "resolution": spec["label"], "width": width, "height": height, "target_fps": fps,
                            "bitrate_target": spec["bitrate"], "encoder": "", "encoder_name": "", "gpu": False,
                            "fallback_reason": ""}
    progress: deque = deque()          # (perf_counter, encoded frame count)
    history: list[dict[str, float]] = []
    state = {"bytes": 0, "frames": 0, "submitted": 0, "dropped": 0, "measure_t": None, "measure_f": 0,
             "measure_bytes": 0, "measure_submitted": 0, "measure_dropped": 0, "stderr": deque(maxlen=20)}
    stats = _Stats(np)

    def rate_between(t_from):
        pts = [p for p in progress if p[0] >= t_from]
        if len(pts) < 2 or pts[-1][0] <= pts[0][0]:
            return 0.0
        return (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0])

    def snapshot(final=False):
        d = dict(info)
        now = pc()
        d.update({"frames_encoded": state["frames"], "frames_submitted": state["submitted"],
                  "frames_dropped": state["dropped"], "instant_fps": rate_between(now - 5.0), "updated_at": time.time()})
        mt = state["measure_t"]
        if mt is not None and progress and progress[-1][0] > mt:
            span = progress[-1][0] - mt
            enc = state["frames"] - state["measure_f"]
            d["fps"] = enc / span
            d["measurement_seconds"] = span
            d["keep_up_ratio"] = enc / (span * fps)
            d["bitrate_mbps"] = (state["bytes"] - state["measure_bytes"]) * 8 / span / 1e6
            d["dropped_in_window"] = state["dropped"] - state["measure_dropped"]
        else:
            d["fps"] = None
        if stats.count:
            d.update({"write_mean_ms": stats.mean * 1000, "write_max_ms": stats.max * 1000})
            for k, v in stats.percentiles.items():
                d[f"write_{k}_ms"] = v
        if final:
            d["fps_history"] = history
        return d

    def launch(enc):
        cmd = [spec["ffmpeg"], "-hide_banner", "-loglevel", "error", "-nostats", "-progress", "pipe:2",
               "-stats_period", "0.5", *enc[3], "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{width}x{height}",
               "-r", str(fps), "-i", "pipe:0", *enc[4], *_encoder_args(enc[0], enc[2], spec["bitrate"], fps),
               # CPU encoder: libx264 sizes its thread pool to ~1.5 x ALL logical
               # cores per encoder, so N streams would run N x that many
               # threads. Each stream gets its share of the cores instead.
               *(["-threads", str(spec["x264_threads"])] if enc[0] == "libx264" and spec.get("x264_threads") else []),
               "-f", "h264", "pipe:1"]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                bufsize=0, creationflags=_WIN_FLAGS)

        def drain():
            while True:
                chunk = proc.stdout.read(1 << 16)
                if not chunk:
                    return
                state["bytes"] += len(chunk)

        def read_progress():
            for raw in iter(proc.stderr.readline, b""):
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("frame="):
                    try:
                        n = int(line.split("=", 1)[1])
                    except ValueError:
                        continue
                    t = pc()
                    state["frames"] = n
                    progress.append((t, n))
                    while len(progress) > 2 and progress[0][0] < t - 30:
                        progress.popleft()
                    if n > 0 and state.get("first_t") is None:
                        state["first_t"] = t
                    if state["measure_t"] is None and state.get("first_t") and t - state["first_t"] >= WARMUP_SECONDS:
                        state.update(measure_t=t, measure_f=n, measure_bytes=state["bytes"],
                                     measure_submitted=state["submitted"], measure_dropped=state["dropped"])
                        info["state"] = "running"
                elif line and "=" not in line:
                    state["stderr"].append(line)
        threading.Thread(target=drain, daemon=True).start()
        threading.Thread(target=read_progress, daemon=True).start()
        return proc

    if frames is None:
        frames = _frames(np, width, height, 8 if width <= 1920 else 4)
    candidates = ([tuple(spec["gpu_encoder"])] if spec.get("gpu_encoder") else []) + [CPU_ENCODER]
    proc = None
    for i, enc in enumerate(candidates):
        info.update(encoder=enc[1], encoder_name=enc[0], gpu=enc[0] != "libx264", state="warming_up")
        proc = launch(enc)
        ok = True
        t_end = pc() + 3.0
        k = 0
        # Prove the encoder works (frames come out) before committing to it.
        while pc() < t_end and state["frames"] < 5:
            try:
                proc.stdin.write(frames[k % len(frames)]); k += 1
            except (BrokenPipeError, OSError):
                ok = False
                break
            if proc.poll() is not None:
                ok = False
                break
            time.sleep(1.0 / fps)
        if ok and state["frames"] > 0:
            break
        err = " | ".join(list(state["stderr"])[-3:]) or f"exit code {proc.poll()}"
        try:
            proc.kill()
        except Exception:
            pass
        if i + 1 < len(candidates):
            info["fallback_reason"] = f"{enc[1]} failed to start: {err[:300]}"
            state.update(frames=0, bytes=0, first_t=None, measure_t=None)
            progress.clear()
        else:
            info.update(state="failed", error=f"{enc[1]} failed: {err[:300]}")
            publish_json(out_file, snapshot(final=True))
            if high_res:
                _high_res_timer(False)
            return

    publish_json(out_file, snapshot())
    period = 1.0 / fps
    next_due = pc()
    next_publish = pc() + 1.0
    next_pct = pc() + 10.0
    k = 0
    while not local_stop.is_set():
        now = pc()
        delay = next_due - now
        if delay > 0:
            if local_stop.wait(delay):
                break
        elif delay < -period:
            # A live camera does not wait for the encoder: slots that passed
            # while the encoder was back-pressuring are dropped.
            missed = int(-delay / period)
            state["dropped"] += missed
            next_due += missed * period
        next_due += period
        t0 = pc()
        try:
            proc.stdin.write(frames[k % len(frames)])
        except (BrokenPipeError, OSError) as exc:
            info.update(state="failed", error=f"encoder stopped: {exc}; " + " | ".join(list(state["stderr"])[-3:]))
            break
        t1 = pc()
        k += 1
        state["submitted"] += 1
        if state["measure_t"] is not None:
            stats.add(t0, t1)
        if t1 >= next_publish:
            if t1 >= next_pct:
                stats.refresh_percentiles()
                next_pct = t1 + 10.0
            history.append({"elapsed_seconds": time.time() - session_t0, "fps": rate_between(t1 - 5.0)})
            publish_json(out_file, snapshot())
            next_publish = t1 + 1.0

    # Let the encoder flush, so the final frame count is the true total.
    try:
        proc.stdin.close()
    except Exception:
        pass
    try:
        proc.wait(10)
    except Exception:
        proc.kill()
    time.sleep(0.2)
    stats.refresh_percentiles()
    if info["state"] != "failed":
        info["state"] = "stopped"
    publish_json(out_file, snapshot(final=True))
    if high_res:
        _high_res_timer(False)


# --------------------------------------------------------------------------- #
# Parent side
# --------------------------------------------------------------------------- #
class EncodeWorkload:
    def __init__(self, cfg: dict[str, Any] | None, out_dir: Path, session_t0: float):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        key = str(cfg.get("resolution", "1080p30")).lower()
        self.preset = dict(PRESETS.get(key, PRESETS["1080p30"]))
        self.preset_key = key if key in PRESETS else "1080p30"
        self.count = max(1, int(cfg.get("stream_count", 1) or 1))
        self.worker_mode = str(cfg.get("worker_mode", "shared")).lower()
        self.out_dir = Path(out_dir)
        self.session_t0 = session_t0
        self.procs: list[tuple[str, multiprocessing.Process, Any, Path]] = []
        self.probe_notes: list[str] = []
        self.error = ""
        self.gpu_encoder = None
        self.ffmpeg = None
        self._last_good: dict[str, dict] = {}

    def start(self) -> None:
        if not self.enabled:
            return
        candidates = ffmpeg_candidates()
        if not candidates:
            self.error = "FFmpeg not found: conda install -c conda-forge ffmpeg, or pip install imageio-ffmpeg (or set VX3_FFMPEG)"
            print(f"[encode] {self.error}", flush=True)
            return
        p = self.preset
        # Different FFmpeg builds ship different hardware encoders, so every
        # FFmpeg found is probed and the first one with a WORKING GPU encoder
        # is used; otherwise the first one (CPU libx264).
        ffmpeg, self.gpu_encoder, self.probe_notes = candidates[0], None, []
        for cand in candidates:
            enc, notes = probe_gpu_encoder(cand, p["width"], p["height"], p["fps"])
            self.probe_notes += [f"{Path(cand).name}: {n}" for n in notes]
            if enc:
                ffmpeg, self.gpu_encoder = cand, enc
                break
        self.ffmpeg = ffmpeg
        chosen = self.gpu_encoder[1] if self.gpu_encoder else CPU_ENCODER[1]
        print(f"[encode] {self.count} x {p['label']} stream(s); encoder: {chosen} ({'; '.join(self.probe_notes)})", flush=True)
        spec = {**p, "ffmpeg": ffmpeg, "gpu_encoder": list(self.gpu_encoder) if self.gpu_encoder else None,
                "x264_threads": max(1, (os.cpu_count() or 1) // self.count)}
        if self.worker_mode == "shared":
            names = [f"encode-{i + 1}" for i in range(self.count)]
            files = [self.out_dir / "aux" / f"{n}.json" for n in names]
            files[0].parent.mkdir(parents=True, exist_ok=True)
            stop = multiprocessing.Event()
            host = multiprocessing.Process(target=_encode_host, args=(names, spec, [str(f) for f in files], self.session_t0, stop),
                                           name="encode-host", daemon=True)
            host.start()
            self.procs = [(n, host, stop, f) for n, f in zip(names, files)]
            return
        for i in range(self.count):
            name = f"encode-{i + 1}"
            out_file = self.out_dir / "aux" / f"{name}.json"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            stop = multiprocessing.Event()
            proc = multiprocessing.Process(target=_encode_stream_process, args=(name, spec, str(out_file), self.session_t0, stop),
                                           name=name, daemon=True)
            proc.start()
            self.procs.append((name, proc, stop, out_file))

    def snapshot(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        streams = []
        for name, proc, _, out_file in self.procs:
            try:
                d = json.loads(out_file.read_text(encoding="utf-8"))
                self._last_good[name] = d
            except Exception:
                d = dict(self._last_good.get(name) or {"name": name, "state": "starting", "fps": None, "instant_fps": 0.0})
            if proc.exitcode not in (None, 0) and d.get("state") not in ("stopped", "failed"):
                d.update(state="failed", error=d.get("error") or f"process exited with code {proc.exitcode}")
            streams.append(d)
        running = [s for s in streams if s.get("fps") is not None]
        return {
            "resolution": self.preset["label"], "preset": self.preset_key, "stream_count": self.count,
            "target_fps": self.preset["fps"], "gpu_encoder": self.gpu_encoder[1] if self.gpu_encoder else None,
            "probe": self.probe_notes, "error": self.error, "ffmpeg": self.ffmpeg, "worker_mode": self.worker_mode,
            "x264_threads_per_stream": max(1, (os.cpu_count() or 1) // self.count),
            "total_fps": sum(float(s.get("fps") or 0) for s in running),
            "total_instant_fps": sum(float(s.get("instant_fps") or 0) for s in streams),
            "streams_keeping_up": sum(1 for s in running if (s.get("keep_up_ratio") or 0) >= 0.98),
            "streams": streams,
        }

    def stop(self) -> None:
        for _, _, stop, _ in self.procs:
            stop.set()

    def join(self, timeout: float = 20) -> None:
        for _, proc, _, _ in {id(p[1]): p for p in self.procs}.values():
            proc.join(timeout)
            if proc.is_alive():
                proc.terminate()
                proc.join(2)
