"""Stream capture, plus an optional, fully detached decode stage.

Two independent stages, each in its own thread, each always working on
whichever frame is currently latest rather than draining a backlog:

1. Capture: reads frames from the source as fast as it provides them, and
   records when each one arrived. This is the ONLY place the main FPS
   number is measured -- nothing downstream can throttle it, because the
   decode stage below never runs inline with it, and never even makes the
   capture loop wait on it (see _LatestFrameBox.put below).
2. Decode: does the color conversion and resize this tool would do if it
   actually needed a usable image -- work that main capture intentionally
   skips (see sources.py). It runs fully detached: it pulls whichever
   captured frame is currently latest whenever it's ready for another one,
   never waits on capture, and never slows it down. Its own FPS is tracked
   completely separately, so a slow decode step only ever shows up as a
   lower decode FPS -- never as a lower main FPS.

Decode can optionally run on a GPU instead of the CPU, via OpenCV's OpenCL
Transparent API (cv2.UMat) -- this works with the standard pip
opencv-python package across NVIDIA/AMD/Intel GPUs, including integrated
ones, with no special build required. It's opt-in (`decode_device: "gpu"`
or `"auto"`) and never silently claims GPU use it didn't get: if OpenCL
isn't actually available, it falls back to CPU and says so in both the
engine log and metrics.decode_device, so a report always states what
genuinely ran, not what was merely requested.
"""
from __future__ import annotations

import threading


class _LatestFrameBox:
    """Single-slot handoff between capture and decode: holds only the most
    recently captured frame. Decode always works on whatever is currently
    latest rather than draining a queue, so a slow decode step never builds
    up backlog/latency -- it just ends up processing a smaller fraction of
    the frames capture produced.

    put() is a plain reference swap -- O(1) regardless of frame size, so it
    never costs the capture loop anything measurable. The copy needed for
    safe cross-thread handoff happens in wait_next() instead, on the decode
    side, and outside the lock -- so it can never block put() either, no
    matter how large the frame or how long the copy takes."""

    def __init__(self):
        self._cond = threading.Condition()
        self._frame = None
        self._captured_at = None
        self._seq = 0

    def put(self, frame, captured_at):
        with self._cond:
            self._frame = frame
            self._captured_at = captured_at
            self._seq += 1
            self._cond.notify_all()

    def wait_next(self, last_seq, timeout=1.0):
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            if self._seq == last_seq or self._frame is None:
                return None, None, last_seq
            frame_ref = self._frame
            captured_at = self._captured_at
            seq = self._seq
        # Copy OUTSIDE the lock: this can take real time for a large frame,
        # and must never make put() (called from the capture thread) block
        # waiting for it. A concurrent put() simply replaces self._frame
        # with a new reference -- it can't affect the object frame_ref
        # already points to here.
        frame = frame_ref.copy()
        return frame, captured_at, seq


def _capture_loop(source, metrics, stop_event, box):
    while not stop_event.is_set():
        result = source.read()
        if result is None:
            metrics.inc("dropped")
            continue
        frame, captured_at = result
        if frame is None:
            metrics.inc("dropped")
            continue
        height, width = frame.shape[:2]
        metrics.set(width=width, height=height)
        # Main FPS is measured right here, before anything else touches
        # this frame. Everything after this line -- including handing the
        # size: box.put() is an O(1) reference swap, never a copy.
        metrics.note_frame(captured_at)
        box.put(frame, captured_at)

def run_stream(source, metrics, stop_event) -> None:
    source.open()
    metrics.set(state="running")

    box = _LatestFrameBox()
    local_stop = threading.Event()

    try:
        _capture_loop(source, metrics, stop_event, box)
    finally:
        local_stop.set()
        source.close()
        metrics.set(state="stopped")
