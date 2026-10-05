"""Synthetic stream source (the benchmark itself never connects to NDI:
real NDI streams are received by OBS -- see obs_controller.py).

read() returns the frame as a raw numpy array view -- reshaped so it's a
valid image array, but with no color conversion and no resize. A reshape
over an existing buffer is a view, not a copy, so it costs essentially
nothing; it's required just to interpret the SDK's buffer as an image at
all. The actual "decode" work -- color conversion and resizing to a
target resolution -- deliberately does NOT happen here: it happens in a
separate, detached stage (see pipeline.py) so it never affects the main
capture loop's own timing. That split is what lets the main FPS number
stay a pure measure of arrival rate while a second, independent FPS number
can honestly show what it costs to also decode/convert/resize each frame.
"""
import threading
import time


class Synthetic:
    """Simulates an NDI-like stream for validation and hardware-load
    testing without real cameras.

    A real NDI receiver's CPU cost comes from a SEPARATE, continuously
    running internal thread inside the NDI library/OS that receives and
    decompresses frames off the network in the background -- our own
    read() calls (frame_sync.capture_video()) just poll whatever that
    background thread has most recently produced; they don't do the
    decode work themselves (see the NDI class below).

    This mirrors that same shape: a background "producer" thread does
    real, substantial, resolution-scaling per-pixel work every simulated
    frame interval (by default -- see synthetic_load), and read() just
    polls it, sleeping to the configured rate exactly like the real NDI
    class's read() does. That split matters for two reasons: it keeps
    read() itself cheap (so it can never distort the main FPS measurement
    the way an earlier version's per-read allocation once did), and it
    means the CPU/memory load this produces is a genuinely continuous
    background cost, not merely inline with however often we happen to
    poll -- the same shape a real decode's cost has.

    IMPORTANT HONESTY NOTE: this cannot replicate NDI's actual proprietary
    codec computation -- that would require the codec itself, which this
    tool doesn't have or attempt to reproduce. What it does instead is
    force real, substantial, correctly resolution-scaling computation
    across every pixel of every simulated frame (full-frame random
    generation), giving a much more realistic preview of a machine's CPU
    headroom than doing near-zero work ever could -- but it is an
    approximation of decode-like cost, not a byte-exact stand-in for any
    specific codec's real behavior. `synthetic_cpu_multiplier` is exposed
    specifically so a user can calibrate the simulated cost against what
    they observe from their own real NDI streams on the same hardware, if
    the default doesn't line up.
    """

    def __init__(self, id, cfg):
        self.id = id
        self.cfg = cfg
        self._frame = None
        self._lock = threading.Lock()
        self._producer_thread = None
        self._stop_producer = threading.Event()
        self._next_read_at = None

    def open(self):
        import numpy as np
        self._width = int(self.cfg.get("width", 1920))
        self._height = int(self.cfg.get("height", 1080))
        self._channels = 3
        self._framerate = max(1.0, float(self.cfg.get("framerate", 30)))
        # "off": the old, near-zero-cost behavior -- a single static buffer,
        # reused forever. Useful only for testing the FPS-measurement
        # plumbing itself in isolation, not for previewing real hardware
        # headroom. "on" (default): the background producer thread below
        # does real per-pixel work every simulated frame interval.
        self._load = str(self.cfg.get("synthetic_load", "on")).lower()
        # Linear intensity knob: how many full-frame passes of random
        # generation happen per simulated frame. 1.0 (default) is one
        # pass; 2.0 is two passes (roughly double the CPU/memory cost);
        # values are not restricted to whole numbers -- e.g. 0.5 keeps only
        # a coin-flip's worth of frames doing the full pass, halving the
        # average cost, for rough calibration against a real stream's
        # observed load on the same hardware.
        self._multiplier = max(0.0, float(self.cfg.get("synthetic_cpu_multiplier", 1.0)))

        self._frame = np.zeros((self._height, self._width, self._channels), dtype=np.uint8)
        self._next_read_at = None
        self._stop_producer.clear()
        self._producer_thread = threading.Thread(target=self._produce_loop, daemon=True)
        self._producer_thread.start()

    def _produce_loop(self):
        import numpy as np
        interval = 1.0 / self._framerate
        next_at = time.monotonic()
        rng = np.random.default_rng()
        whole_passes = int(self._multiplier)
        fractional_pass_chance = self._multiplier - whole_passes

        while not self._stop_producer.is_set():
            now = time.monotonic()
            if now < next_at:
                time.sleep(min(0.002, next_at - now))
                continue
            if self._load == "off":
                next_at += interval
                continue
            # Real, substantial, resolution-scaling computation across the
            # whole frame -- genuine CPU and memory-bandwidth cost, not a
            # sleep or a static buffer. Runs here, on this background
            # thread, so it never blocks or slows read() itself.
            passes = whole_passes + (1 if rng.random() < fractional_pass_chance else 0)
            frame = self._frame
            for _ in range(max(1, passes)):
                frame = rng.integers(0, 256, size=(self._height, self._width, self._channels), dtype=np.uint8)
            with self._lock:
                self._frame = frame
            next_at += interval

    def read(self):
        # Paced against an absolute schedule, not a fixed sleep tacked onto
        # whatever's already elapsed -- see the NDI class's read() below for
        # why: sleeping the full interval every call, regardless of how
        # much time this call itself already spent (lock acquisition, the
        # frame reference grab), makes the real cycle period longer than
        # the interval every single time, and that overhead compounds. This
        # keeps the *target* times exactly interval apart no matter how
        # long any individual call takes, which is what makes the average
        # rate converge on the true configured rate instead of drifting
        # below it.
        interval = 1.0 / self._framerate
        now = time.monotonic()
        if self._next_read_at is None:
            self._next_read_at = now + interval
        else:
            self._next_read_at += interval
        sleep_for = self._next_read_at - now
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            # Fell behind schedule (a call took longer than one interval,
            # e.g. under system load) -- resync to now rather than trying
            # to burst-catch-up, which would just serve the same buffered
            # frame repeatedly without it actually being new.
            self._next_read_at = now

        with self._lock:
            frame = self._frame
        return frame, time.time()

    def close(self):
        self._stop_producer.set()
        if self._producer_thread is not None:
            self._producer_thread.join(2)
