"""Per-stream subprocess entry point for SYNTHETIC test streams.

Real NDI streams are never received by the benchmark: OBS receives them
(see obs_controller.py). A synthetic stream runs in its own OS process and
publishes its metrics to a small JSON file.
"""
from __future__ import annotations

import sys
import threading
import traceback
from pathlib import Path
from typing import Any

from .live_metrics import Metrics, MetricsPublisher, stream_snapshot_path, stream_history_path
from .pipeline import run_stream
from .sources import Synthetic


def _begin_high_res_timer() -> bool:
    """On Windows, time.sleep() defaults to the system's ordinary timer
    granularity -- commonly around 15.6ms -- which can silently add several
    extra milliseconds to every single read() call's sleep, on top of (and
    separate from) correct scheduling. Real-time/multimedia Windows
    applications routinely request a higher-resolution timer for exactly
    this reason. This is a per-process setting, requested once here for
    the whole lifetime of this stream's process. No-op, silently, on any
    other platform, or if the call isn't available for any reason -- this
    is a precision improvement, never something capture should depend on
    to function."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
        return True
    except Exception:
        return False


def _end_high_res_timer() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.winmm.timeEndPeriod(1)
    except Exception:
        pass


def run_stream_process(spec: dict[str, Any], stop_event) -> None:
    """One synthetic stream in its own OS process."""
    high_res_timer = _begin_high_res_timer()
    try:
        _run_one_stream(spec, stop_event)
    finally:
        if high_res_timer:
            _end_high_res_timer()


def _run_one_stream(spec: dict[str, Any], stop_event) -> None:
    """Everything one stream needs, built from scratch. `spec` is a plain,
    picklable dict -- nothing unpicklable ever
    crosses the process boundary; only small JSON-safe config does, at
    start, plus a small JSON snapshot file throughout the run."""
    stream_id = spec["stream_id"]
    transport = spec["transport"]
    out_dir = Path(spec["out_dir"])
    metrics = Metrics(stream_id, transport, session_t0=spec.get("session_t0"),
                      history_interval=spec.get("fps_history_interval_seconds", 1.0))

    snapshot_path = stream_snapshot_path(out_dir, stream_id)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    # A LOCAL stop signal for the publisher thread only -- distinct from the
    # shared `stop_event` that governs when capture itself should stop. A
    # failure in this one stream must not touch the shared stop_event.
    local_stop = threading.Event()
    publisher = MetricsPublisher(metrics, snapshot_path, local_stop, interval=0.5,
                                 history_path=stream_history_path(out_dir, stream_id))
    publisher.start()
    try:
        source = Synthetic(stream_id, spec["source_cfg"])
        run_stream(source, metrics, stop_event)
    except Exception as exc:
        metrics.set(state="failed", last_error=str(exc))
        traceback.print_exc()
    finally:
        local_stop.set()
        publisher.join(3)
