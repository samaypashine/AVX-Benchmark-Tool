"""Thread-safe per-stream metrics: how many frames arrived, and how fast.

FPS history
-----------
The report's "FPS over time" chart needs the WHOLE session, for every
stream. The earlier implementation appended one history entry per frame and
only ever published the last 1800 of them -- at 30 fps that is the final
60 seconds of the run, which is why the chart looked broken on any real
session (and the per-frame list grew without bound in memory).

History is now sampled on a fixed cadence (default 1 s) by the publisher
thread, not by the capture loop:

* each point is the exact frame rate over that interval
  (frames received during the interval / monotonic interval length), so a
  stall correctly shows as 0 fps instead of simply producing no points;
* elapsed time is measured from the shared session start, so every stream's
  curve lines up on the same time axis;
* the list is bounded: once it reaches ``history_cap`` points, adjacent
  pairs are merged (keeping the interval's mean plus its min and max), so a
  24 h+ soak stays small without losing dips.

Nothing here runs in the capture loop except the existing counter update in
note_frame(), so the capture/receive measurement itself is unaffected.
"""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import threading
import time
from typing import Any

from .config import slug
from .live_publish import publish_json


class Metrics:
    def __init__(self, stream_id: str, transport: str, fps_window_seconds: float = 5.0,
                 session_t0: float | None = None, history_interval: float = 1.0,
                 history_cap: int = 20000):
        self.stream_id = stream_id
        self.transport = transport
        self.started = time.time()
        self.session_t0 = float(session_t0) if session_t0 else self.started
        self.fps_window_seconds = max(1.0, float(fps_window_seconds))
        self.lock = threading.Lock()
        self.frame_times: deque[float] = deque(maxlen=3600)
        self._first_frame_at: float | None = None
        # --- whole-session history (see module docstring) ---
        self.history_interval = max(0.1, float(history_interval))
        self.history_cap = max(1000, int(history_cap))
        self._history: list[dict[str, float]] = []
        self._last_sample_mono: float | None = None
        self._last_sample_frames = 0
        self._sample_count = 0
        self._sample_sum = 0.0
        self._sample_min: float | None = None
        self._sample_max: float | None = None
        self.values: dict[str, Any] = {
            "stream_id": stream_id,
            "transport": transport,
            "state": "starting",
            "frames": 0,
            "dropped": 0,
            "width": None,
            "height": None,
            "last_error": "",
        }

    def inc(self, key: str, value: int | float = 1) -> None:
        with self.lock:
            self.values[key] = self.values.get(key, 0) + value

    def set(self, **values: Any) -> None:
        with self.lock:
            self.values.update(values)

    def _rolling_fps(self, times: deque[float], now: float) -> float:
        cutoff = now - self.fps_window_seconds
        while times and times[0] < cutoff:
            times.popleft()
        if len(times) < 2:
            return 0.0
        span = max(0.001, times[-1] - times[0])
        return (len(times) - 1) / span

    def note_frame(self, timestamp: float | None = None) -> float:
        """Records one arrived frame (the main capture rate) and updates its
        rolling FPS window."""
        now = float(timestamp or time.time())
        with self.lock:
            if self._first_frame_at is None:
                self._first_frame_at = now
            self.values["frames"] += 1
            self.frame_times.append(now)
            fps = self._rolling_fps(self.frame_times, now)
            self.values["instant_fps"] = fps
            return fps

    # ------------------------------------------------------------------ #
    def sample_history(self, final: bool = False) -> None:
        """Called by the publisher thread about once per history_interval."""
        mono = time.monotonic()
        wall = time.time()
        with self.lock:
            frames = self.values["frames"]
            if self._last_sample_mono is None or self._first_frame_at is None or frames == 0:
                # Baseline only, and it is only taken once frames are already
                # flowing, so the first point never mixes connect/discovery
                # time into its rate.
                self._last_sample_mono = mono if frames else None
                self._last_sample_frames = frames
                return
            interval = mono - self._last_sample_mono
            if interval <= 0 or (final and interval < 0.5 * self.history_interval):
                return  # a sliver of an interval at shutdown is not a real sample
            fps = (frames - self._last_sample_frames) / interval
            self._last_sample_mono = mono
            self._last_sample_frames = frames
            self._history.append({"elapsed_seconds": round(wall - self.session_t0, 3),
                                  "fps": fps, "fps_min": fps, "fps_max": fps,
                                  "interval_seconds": interval})
            self._sample_count += 1
            self._sample_sum += fps
            self._sample_min = fps if self._sample_min is None else min(self._sample_min, fps)
            self._sample_max = fps if self._sample_max is None else max(self._sample_max, fps)
            if len(self._history) > self.history_cap:
                self._compact()

    def _compact(self) -> None:
        """Merge adjacent pairs: time-weighted mean, envelope min/max."""
        merged = []
        h = self._history
        for i in range(0, len(h) - 1, 2):
            a, b = h[i], h[i + 1]
            span = a["interval_seconds"] + b["interval_seconds"]
            merged.append({
                "elapsed_seconds": b["elapsed_seconds"],
                "fps": (a["fps"] * a["interval_seconds"] + b["fps"] * b["interval_seconds"]) / span,
                "fps_min": min(a["fps_min"], b["fps_min"]),
                "fps_max": max(a["fps_max"], b["fps_max"]),
                "interval_seconds": span,
            })
        if len(h) % 2:
            merged.append(h[-1])
        self._history = merged

    def history(self) -> list[dict[str, float]]:
        with self.lock:
            return [dict(x) for x in self._history]

    def snapshot(self, include_history: bool = False) -> dict[str, Any]:
        with self.lock:
            output = dict(self.values)
            first_frame_at = self._first_frame_at
            count = self._sample_count
            if count:
                output["fps_min"] = self._sample_min
                output["fps_mean"] = self._sample_sum / count
                output["fps_max"] = self._sample_max
            output["history_points"] = len(self._history)
            if include_history:
                output["fps_history"] = [dict(x) for x in self._history]

        now = time.time()
        elapsed = max(0.001, now - (first_frame_at if first_frame_at is not None else self.started))
        output["active_seconds"] = elapsed
        output["fps"] = output["frames"] / elapsed
        return output


def stream_snapshot_path(out_dir, stream_id) -> Path:
    """Where a stream's live Metrics snapshot is published, and read back by
    engine.run()'s aggregation loop -- this file is the one thing that has
    to cross a process boundary since each stream runs in its own OS
    process, so it's a plain small JSON file rather than anything requiring
    shared memory or a multiprocessing manager."""
    return Path(out_dir) / "streams" / f"{slug(stream_id)}.json"


def stream_history_path(out_dir, stream_id) -> Path:
    """Full-session FPS history for one stream (kept out of the live
    snapshot so the 2 Hz live publish stays tiny)."""
    return Path(out_dir) / "streams" / f"{slug(stream_id)}.history.json"


def read_stream_history(out_dir, stream_id) -> list[dict[str, float]]:
    try:
        data = json.loads(stream_history_path(out_dir, stream_id).read_text(encoding="utf-8"))
        return data.get("fps_history", []) if isinstance(data, dict) else []
    except Exception:
        return []


class MetricsPublisher(threading.Thread):
    """Periodically publishes a Metrics object's snapshot to its per-stream
    JSON file, samples the FPS history on a fixed cadence, and checkpoints
    that history to its own file (so a stream process that is killed still
    leaves almost all of its history behind)."""

    def __init__(self, metrics: Metrics, out_path: Path, stop_event, interval: float = 0.5,
                 history_path: Path | None = None, history_checkpoint_seconds: float = 30.0):
        super().__init__(daemon=True)
        self.metrics = metrics
        self.out_path = out_path
        self.history_path = history_path
        self.stop_event = stop_event
        self.interval = interval
        self.history_checkpoint_seconds = max(5.0, float(history_checkpoint_seconds))

    def _write_history(self) -> None:
        if self.history_path is not None:
            publish_json(self.history_path, {"stream_id": self.metrics.stream_id,
                                             "fps_history": self.metrics.history()})

    def run(self) -> None:
        start = time.monotonic()
        next_publish = start
        next_sample = start
        next_checkpoint = start + self.history_checkpoint_seconds
        while True:
            now = time.monotonic()
            if now >= next_sample:
                self.metrics.sample_history()
                next_sample += self.metrics.history_interval
                if next_sample < now:          # fell behind (e.g. system stall)
                    next_sample = now + self.metrics.history_interval
            if now >= next_publish:
                publish_json(self.out_path, self.metrics.snapshot())
                next_publish = now + self.interval
            if now >= next_checkpoint:
                self._write_history()
                next_checkpoint = now + self.history_checkpoint_seconds
            wait = max(0.0, min(next_sample, next_publish) - time.monotonic())
            if self.stop_event.wait(wait):
                break
        # Final publish so the last thing on disk reflects the stream's true
        # end state (e.g. state: "stopped"/"failed") and its full history.
        self.metrics.sample_history(final=True)
        self._write_history()
        publish_json(self.out_path, self.metrics.snapshot())
