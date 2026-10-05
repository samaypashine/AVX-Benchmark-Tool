"""Independent image workloads measured with the same precision rules as the
AI model workers (see model_workers.py):

(The video encode test lives in encode_workload.py.)
* Image persistence -- ONE separate OS process PER ENABLED AI MODEL that
  repeatedly saves the same ("stale") pre-encoded image to the SAME file
  path, paced to that model's live measured throughput -- i.e. one saved
  image per AI inference, exactly as a production pipeline that persists
  each processed frame would. Until the model is producing inferences the
  process idles, and idle time is excluded from its FPS.

Precision:
* only the work itself is timed with perf_counter (the encode call; or
  open + write + flush + optional fsync + close for persistence);
* pacing uses an absolute schedule (no burst catch-up), so sleep jitter
  averages out;
* FPS = completed operations / measurement window, excluding warm-up,
  publishing and idle time; latency percentiles are exact to 1 us;
* nothing crosses the process boundary per operation -- a small JSON
  snapshot is published about once a second.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
from pathlib import Path
from typing import Any

from .config import slug
from .live_publish import publish_json
from .model_workers import _FPS_WINDOW, _PERCENTILE_INTERVAL, _PUBLISH_INTERVAL, _Stats, _high_res_timer


class _AuxStats(_Stats):
    """Adds idle-time exclusion locally, so this module works with any
    version of model_workers._Stats (an older model_workers.py without the
    'idle' attribute crashed the persistence processes)."""

    def __init__(self, np):
        super().__init__(np)
        self.idle = 0.0

    def fps(self):
        if not self.count or self.window_start is None:
            return None
        busy = (self.last_end - self.window_start) - self.instrumentation - self.idle
        return self.count / busy if busy > 0 else None

FORMATS = ("jpeg", "png", "webp")
DEFAULT_PERSISTENCE = {"enabled": False, "directory": "", "width": 1920, "height": 1080, "format": "jpeg",
                       "quality": 90, "fsync": True}


# --------------------------------------------------------------------------- #
# Image helpers (child side)
# --------------------------------------------------------------------------- #
def _synthetic_image(np, width: int, height: int):
    """Deterministic test picture: smooth gradients (compress well) plus a
    textured region (compresses poorly), so encode cost resembles a camera
    frame rather than a flat colour."""
    y, x = np.mgrid[0:height, 0:width]
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[..., 0] = (x * 255 // max(1, width - 1)).astype(np.uint8)
    img[..., 1] = (y * 255 // max(1, height - 1)).astype(np.uint8)
    img[..., 2] = ((x + y) * 255 // max(1, width + height - 2)).astype(np.uint8)
    rng = np.random.default_rng(1234)
    h2, w2 = height // 3, width // 3
    img[h2:2 * h2, w2:2 * w2] = rng.integers(0, 256, size=(h2, w2, 3), dtype=np.uint8)
    return img


def _encoder(fmt: str, quality: int):
    """Returns (encode(img) -> bytes, backend name). OpenCV releases the GIL
    and is the fastest; Pillow is the fallback."""
    fmt = fmt.lower()
    if fmt not in FORMATS:
        raise ValueError(f"unsupported format {fmt!r}; use one of {', '.join(FORMATS)}")
    quality = max(1, min(100, int(quality)))
    try:
        import cv2
        ext = {"jpeg": ".jpg", "png": ".png", "webp": ".webp"}[fmt]
        params = {"jpeg": [cv2.IMWRITE_JPEG_QUALITY, quality],
                  "png": [cv2.IMWRITE_PNG_COMPRESSION, 3],
                  "webp": [cv2.IMWRITE_WEBP_QUALITY, quality]}[fmt]

        def encode(img):
            ok, buf = cv2.imencode(ext, img, params)
            if not ok:
                raise RuntimeError(f"cv2.imencode({ext}) failed")
            return buf.tobytes()
        return encode, f"OpenCV {cv2.__version__}"
    except ImportError:
        import io
        from PIL import Image
        pil_fmt = {"jpeg": "JPEG", "png": "PNG", "webp": "WEBP"}[fmt]
        opts = {"quality": quality} if fmt in ("jpeg", "webp") else {"compress_level": 3}

        def encode(img):
            buf = io.BytesIO()
            Image.fromarray(img[..., ::-1]).save(buf, pil_fmt, **opts)
            return buf.getvalue()
        import PIL
        return encode, f"Pillow {PIL.__version__}"


# --------------------------------------------------------------------------- #
# Generic paced, measured loop (child side)
# --------------------------------------------------------------------------- #
def _paced_loop(name: str, out_file: str, session_t0: float, stop_event, rate_value, setup,
                warmup: int, idle_without_rate: bool, extra_info: dict[str, Any],
                backlog_mode: bool = False, max_backlog_seconds: float = 2.0) -> None:
    """backlog_mode (used by persistence): work is REQUEST-driven. Requests
    accrue continuously at `rate` into a queue and are served as fast as the
    operation allows, so a slow save (fsync spike, antivirus scan, scheduler
    delay under CPU load) is caught up afterwards -- exactly like a real
    pipeline that queues one save per inference. Only requests older than
    max_backlog_seconds are dropped (and counted). The fixed-schedule mode
    used before silently discarded the slot of any save that ran more than
    one frame interval late (6.7 ms at 149 FPS), under-reporting what the
    disk could actually sustain."""
    pc = time.perf_counter
    local_stop = threading.Event()

    def _watch():
        # Poll, never block in stop_event.wait(): a process that exits while
        # registered as a waiter would deadlock the parent's stop_event.set().
        while not stop_event.is_set():
            if local_stop.wait(0.05):
                return
        local_stop.set()
    threading.Thread(target=_watch, daemon=True).start()
    high_res = _high_res_timer(True)

    info: dict[str, Any] = {"name": name, "state": "starting", "error": "", "operations": 0, "failures": 0,
                            "warmup_iterations": warmup, **extra_info}
    history: list[dict[str, float]] = []
    stats = None
    counters = {"bytes": 0, "requested": 0.0, "dropped": 0.0, "backlog": 0.0, "max_backlog": 0.0}
    rate_now = [0.0]

    def snapshot(final=False):
        data = dict(info)
        now = pc()
        if stats is not None and stats.count:
            count = stats.count
            data.update({
                "operations": count, "fps": stats.fps(), "instant_fps": stats.instant_fps(now),
                "latency_mean_ms": stats.mean * 1000, "latency_min_ms": stats.min * 1000,
                "latency_max_ms": stats.max * 1000,
                "latency_stdev_ms": (stats.m2 / (count - 1)) ** 0.5 * 1000 if count > 1 else None,
                "capacity_fps": 1.0 / stats.mean if stats.mean > 0 else None,
                "bytes_per_operation": counters["bytes"] / count,
                "throughput_mb_per_s": (counters["bytes"] / 1e6) * (stats.fps() or 0) / count if count else None,
                "measurement_seconds": stats.last_end - stats.window_start,
                "idle_seconds": stats.idle, "instrumentation_seconds": stats.instrumentation,
            })
            for label, value in stats.percentiles.items():
                data[f"latency_{label}_ms"] = value
            if counters["requested"] > 0:
                data["requested_operations"] = counters["requested"]
                # Requests still waiting in the queue (< 1 is a partial one)
                # are not failures; only served vs. requested-and-due counts.
                due = max(1.0, counters["requested"] - max(0.0, counters["backlog"]))
                data["keep_up_ratio"] = min(count / due, 10.0)
                # Average requested rate over the same active window as fps.
                if stats.fps():
                    data["requested_fps_avg"] = counters["requested"] * stats.fps() / count
            if backlog_mode:
                data.update({"dropped_requests": int(counters["dropped"]), "backlog_now": counters["backlog"],
                             "max_backlog": counters["max_backlog"], "max_backlog_seconds": max_backlog_seconds})
        else:
            data.update({"fps": None, "instant_fps": 0.0, "latency_mean_ms": None})
        data["requested_fps"] = rate_now[0] or None
        data["updated_at"] = time.time()
        if final:
            data["fps_history"] = history
        return data

    try:
        publish_json(out_file, snapshot())
        import numpy as np
        step, more = setup(np)
        info.update(more)
        stats = _AuxStats(np)
        info["state"] = "warming_up" if warmup else "running"
        publish_json(out_file, snapshot())
    except Exception as exc:
        info.update(state="failed", error=str(exc))
        publish_json(out_file, snapshot(final=True))
        local_stop.set()
        if high_res:
            _high_res_timer(False)
        return

    warmed = 0
    measuring = warmup == 0
    next_due = None
    period_now = None
    last_loop = pc()
    next_publish = pc() + _PUBLISH_INTERVAL
    next_pct = pc() + _PERCENTILE_INTERVAL
    while not local_stop.is_set():
        rate = rate_value.value if rate_value is not None else 0.0
        rate_now[0] = rate
        now = pc()
        if backlog_mode:
            elapsed = now - last_loop
            last_loop = now
            if rate > 0:
                arrived = elapsed * rate
                counters["requested"] += arrived
                counters["backlog"] += arrived
                cap = max(1.0, rate * max_backlog_seconds)
                if counters["backlog"] > cap:
                    counters["dropped"] += counters["backlog"] - cap
                    counters["backlog"] = cap
                counters["max_backlog"] = max(counters["max_backlog"], counters["backlog"])
            if counters["backlog"] < 1.0:
                wait = 0.05 if rate <= 0 else min(0.05, max(0.0002, (1.0 - counters["backlog"]) / rate))
                w0 = pc()
                if local_stop.wait(wait):
                    break
                if rate <= 0 and stats.window_start is not None:
                    stats.idle += pc() - w0
                if pc() >= next_publish:
                    publish_json(out_file, snapshot())
                    next_publish = pc() + _PUBLISH_INTERVAL
                continue
            counters["backlog"] -= 1.0
            # fall through to the timed operation below
        if not backlog_mode and rate <= 0 and idle_without_rate:
            # Nothing requested (e.g. the AI model is not producing yet).
            if local_stop.wait(0.05):
                break
            if measuring and stats.window_start is not None:
                stats.idle += pc() - now
            next_due = None
            last_loop = pc()
            if pc() >= next_publish:
                publish_json(out_file, snapshot())
                next_publish = pc() + _PUBLISH_INTERVAL
            continue
        if not backlog_mode and rate > 0:
            period = 1.0 / rate
            if next_due is None:
                next_due = now
            elif period != period_now and next_due > now + period:
                next_due = now + period          # rate rose: don't wait out the old, longer period
            period_now = period
            delay = next_due - now
            if delay > 0:
                if local_stop.wait(delay):
                    break
            elif delay < -period:
                next_due = pc()                  # behind: no catch-up burst
            next_due += period
            if measuring:
                t = pc()
                counters["requested"] += (t - last_loop) * rate
                last_loop = t
        started = pc()
        try:
            produced = step()
            ended = pc()
        except Exception as exc:
            info["failures"] += 1
            info["error"] = str(exc)
            if local_stop.wait(0.1):
                break
            continue
        if not measuring:
            warmed += 1
            if warmed == 1:
                info["cold_start_latency_ms"] = (ended - started) * 1000
            if warmed >= warmup:
                measuring = True
                info["state"] = "running"
                stats.window_start = ended
                last_loop = pc()
            continue
        stats.add(started, ended)
        counters["bytes"] += int(produced or 0)
        if ended >= next_publish:
            b = pc()
            if b >= next_pct:
                stats.refresh_percentiles()
                next_pct = b + _PERCENTILE_INTERVAL
            history.append({"elapsed_seconds": time.time() - session_t0, "fps": stats.instant_fps(b),
                            **({"requested_fps": rate} if rate > 0 else {})})
            publish_json(out_file, snapshot())
            next_publish = b + _PUBLISH_INTERVAL
            stats.instrumentation += pc() - b

    if stats is not None:
        stats.refresh_percentiles()
    info["state"] = "stopped"
    publish_json(out_file, snapshot(final=True))
    if high_res:
        _high_res_timer(False)


def _persist_process(name, cfg, out_file, session_t0, stop_event, rate_value, image_path):
    width, height = int(cfg.get("width", 1920)), int(cfg.get("height", 1080))
    fmt, quality = str(cfg.get("format", "jpeg")), int(cfg.get("quality", 90))
    do_fsync = bool(cfg.get("fsync", True))

    def setup(np):
        encode, backend = _encoder(fmt, quality)
        data = encode(_synthetic_image(np, width, height))   # encoded ONCE: the "stale" image
        Path(image_path).parent.mkdir(parents=True, exist_ok=True)

        def save():
            # The whole persistence operation is timed: open, write, flush,
            # optional fsync (forces it to the storage device instead of
            # the OS cache), close -- always to the same file.
            with open(image_path, "wb") as fh:
                fh.write(data)
                fh.flush()
                if do_fsync:
                    os.fsync(fh.fileno())
            return len(data)
        return save, {"backend": backend, "image_bytes": len(data)}
    _paced_loop(name, out_file, session_t0, stop_event, rate_value, setup, warmup=0, idle_without_rate=True,
                backlog_mode=True, max_backlog_seconds=float(cfg.get("max_backlog_seconds", 2.0)),
                extra_info={"kind": "image_persistence", "width": width, "height": height, "format": fmt,
                            "quality": quality, "fsync": do_fsync, "image_path": str(image_path)})


def _persist_host(entries: list[tuple], host_stop) -> None:
    """Shared mode: ONE OS process running every model's persistence loop
    as a thread. Saving is almost entirely waiting on the disk (file writes
    and fsync release Python's GIL), so threads lose nothing against
    separate processes, and each extra Python/numpy process (~60-100 MB) is
    avoided."""
    local = threading.Event()
    threads = [threading.Thread(target=_persist_process, name=e[0], daemon=True,
                                args=(e[0], e[1], e[2], e[3], local, e[4], e[5])) for e in entries]
    for t in threads:
        t.start()
    while not host_stop.is_set():      # poll only -- never block in Event.wait()
        time.sleep(0.1)
    local.set()
    for t in threads:
        t.join(15)


# --------------------------------------------------------------------------- #
# Parent side
# --------------------------------------------------------------------------- #
class AuxProcess:
    def __init__(self, name: str, target, args: tuple, out_file: Path):
        self.name = name
        self.target = target
        self.args = args
        self.out_file = Path(out_file)
        self.rate = multiprocessing.RawValue("d", 0.0)
        self.stop_event = None
        self.process = None
        self._last_good = None

    def start(self) -> None:
        self.out_file.parent.mkdir(parents=True, exist_ok=True)
        self.stop_event = multiprocessing.Event()
        self.process = multiprocessing.Process(target=self.target, args=(*self.args[:4], self.stop_event, self.rate, *self.args[4:]),
                                               name=self.name, daemon=True)
        self.process.start()

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
        except Exception:
            data = dict(self._last_good) if self._last_good else {"name": self.name, "state": "starting",
                                                                  "fps": None, "instant_fps": 0.0}
        code = self.process.exitcode if self.process is not None else None
        if code not in (None, 0) and data.get("state") not in ("stopped", "failed"):
            data["state"] = "failed"
            data["error"] = data.get("error") or f"process exited with code {code}"
        return data


def _merged(defaults, cfg):
    out = dict(defaults)
    out.update(cfg or {})
    return out


class ImagePersistenceWorkload:
    """One persistence process per enabled AI model, paced to that model's
    live measured throughput (updated every engine tick)."""

    def __init__(self, cfg: dict[str, Any] | None, model_names: list[str], out_dir: Path, session_t0: float):
        self.cfg = _merged(DEFAULT_PERSISTENCE, cfg)
        self.enabled = bool(self.cfg.get("enabled")) and bool(model_names)
        self.worker_mode = str(self.cfg.get("worker_mode", "shared")).lower()
        self.host = None
        directory = str(self.cfg.get("directory") or "").strip()
        base = Path(directory) if directory else Path(out_dir) / "persistence"
        ext = {"jpeg": "jpg", "png": "png", "webp": "webp"}.get(str(self.cfg.get("format", "jpeg")).lower(), "jpg")
        self.procs: dict[str, AuxProcess] = {}
        if self.enabled:
            for name in model_names:
                out_file = Path(out_dir) / "aux" / f"persist-{slug(name)}.json"
                image_path = base / f"{slug(name)}.{ext}"
                self.procs[name] = AuxProcess(f"persist-{name}", _persist_process,
                                              (f"persist:{name}", self.cfg, str(out_file), session_t0, str(image_path)),
                                              out_file)

    def start(self):
        if self.worker_mode == "shared" and self.procs:
            stop = multiprocessing.Event()
            entries = []
            for p in self.procs.values():
                p.out_file.parent.mkdir(parents=True, exist_ok=True)
                name, cfg, out_file, t0, image_path = p.args
                entries.append((name, cfg, out_file, t0, p.rate, image_path))
            self.host = multiprocessing.Process(target=_persist_host, args=(entries, stop), name="persist-host", daemon=True)
            self.host.start()
            for p in self.procs.values():
                p.process, p.stop_event = self.host, stop
            return
        for p in self.procs.values():
            p.start()

    def tick(self, models: list[dict[str, Any]]):
        """Follow each model's measured total throughput."""
        rates = {m.get("name"): float(m.get("measured_fps") if m.get("measured_fps") is not None else (m.get("instant_fps") or 0))
                 for m in models or []}
        for name, p in self.procs.items():
            p.rate.value = max(0.0, rates.get(name, 0.0))

    def snapshot(self):
        out = []
        for name, p in self.procs.items():
            data = p.snapshot()
            data["model"] = name
            data["worker"] = "thread (shared process)" if self.host is not None else "process"
            out.append(data)
        return out

    def stop(self):
        for p in self.procs.values():
            p.rate.value = 0.0
            p.stop()

    def join(self, timeout=10):
        for p in self.procs.values():
            p.join(timeout)
