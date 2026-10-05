#!/usr/bin/env python3
"""VX3 NDI camera emulator -- run on a SECOND PC.

Publishes N NDI sources that behave like real cameras on the network, so the
benchmark PC receives them (in OBS) exactly like its real cameras: same NDI
codec, similar data rate, same network link. Use it to test more streams
than you have real cameras, e.g. 12 real + 8 emulated = 20.

Why a second PC: NDI traffic between two programs on the SAME machine never
goes through the network card, so it cannot reproduce what limits real
streams (link bandwidth, NIC, switch). Only traffic from another machine can.

What it sends
  * moving, detailed test content (gradients, a moving textured band and
    fine noise) so the NDI encoder produces a real-camera data rate --
    unlike flat/black frames, which compress to almost nothing;
  * UYVY frames (the NDI SDK's native format, no conversion on this PC);
  * each camera in its own process, paced on an absolute clock.

Live status every few seconds: per-camera FPS actually sent, number of
receivers connected, and this PC's total network TX. Tune --detail until
the TX per stream matches your cameras (typically 100-125 Mbit/s for
1080p30 full-bandwidth NDI).

Requirements on the sending PC: NDI Runtime, Python 3.10+,
    pip install cyndilib numpy psutil

Examples
    python ndi_camera_emulator.py --streams 8
    python ndi_camera_emulator.py --streams 8 --format 1080p60 --detail 6
    python ndi_camera_emulator.py --streams 4 --format 4k30 --name "LAB EMU"
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import sys
import time

FORMATS = {"720p30": (1280, 720, 30), "720p60": (1280, 720, 60), "1080p25": (1920, 1080, 25),
           "1080p30": (1920, 1080, 30), "1080p50": (1920, 1080, 50), "1080p60": (1920, 1080, 60),
           "4k30": (3840, 2160, 30)}


# --------------------------------------------------------------------------- #
# Test content
# --------------------------------------------------------------------------- #
def make_frames(width: int, height: int, count: int, detail: float, seed: int):
    """`count` UYVY 4:2:2 frames (bytes U Y0 V Y1 per pixel pair).

    detail 0 = smooth content only; each step adds fine noise, which is
    what drives an NDI (SpeedHQ, intra-only) encoder's data rate up the way
    real sensor detail and noise do."""
    import numpy as np
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    band_h = max(8, height // 5)
    # Textured band: stripes and checks with mild grain -- detailed like a
    # real scene, not pure random noise (which would be unrealistically hard
    # to compress and far above a real camera's data rate).
    by, bx = np.mgrid[0:band_h, 0:width]
    texture = (128 + 50 * np.sign(np.sin(bx / 6.0)) * np.sign(np.sin(by / 6.0))
               + 20 * np.sin(bx / 2.3) + rng.normal(0, 6, size=(band_h, width))).clip(16, 235).astype(np.uint8)
    frames = []
    for k in range(count):
        phase = k / max(1, count)
        y = (128 + 60 * np.sin(2 * np.pi * (xx / width + phase)) * np.cos(2 * np.pi * yy / height)).astype(np.float32)
        top = int((height - band_h) * (0.5 + 0.5 * np.sin(2 * np.pi * phase)))
        y[top:top + band_h] = np.roll(texture, int(k * width / count / 2), axis=1)
        if detail > 0:
            y += rng.normal(0, detail * 0.55, size=y.shape).astype(np.float32)
        y = np.clip(y, 16, 235).astype(np.uint8)
        u = np.clip(128 + 40 * np.sin(2 * np.pi * (xx[:, ::2] / width + phase + seed * 0.13))
                    + (rng.normal(0, detail * 0.3, size=(height, width // 2)) if detail > 0 else 0), 16, 240).astype(np.uint8)
        v = np.clip(128 + 40 * np.cos(2 * np.pi * (yy[:, ::2] / height + phase))
                    + (rng.normal(0, detail * 0.3, size=(height, width // 2)) if detail > 0 else 0), 16, 240).astype(np.uint8)
        f = np.empty((height, width * 2), dtype=np.uint8)
        f[:, 0::4] = u
        f[:, 1::4] = y[:, 0::2]
        f[:, 2::4] = v
        f[:, 3::4] = y[:, 1::2]
        frames.append(f.ravel())
    return frames


# --------------------------------------------------------------------------- #
# One emulated camera (own process)
# --------------------------------------------------------------------------- #
def camera(index: int, name: str, width: int, height: int, fps: int, detail: float, bank: int, stats, stop) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)       # the parent handles Ctrl+C
    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass
    from cyndilib.sender import Sender
    from cyndilib.video_frame import VideoSendFrame
    from cyndilib.wrapper.ndi_structs import FourCC

    frames = make_frames(width, height, bank, detail, seed=index)
    sender = Sender(name)
    vf = VideoSendFrame()
    vf.set_resolution(width, height)
    vf.set_fourcc(FourCC.UYVY)
    vf.set_frame_rate(fps)
    sender.set_video_frame(vf)
    sender.open()

    period = 1.0 / fps
    next_due = time.perf_counter()
    sent = 0
    k = index                       # cameras start at different points of the loop
    while not stop.is_set():
        now = time.perf_counter()
        if next_due > now:
            time.sleep(next_due - now)
        elif now - next_due > period:
            next_due = now          # fell behind: resync, never burst
        next_due += period
        sender.write_video(frames[k % bank])
        k += 1
        sent += 1
        stats[index * 2] = sent
        if sent % fps == 0:
            try:
                stats[index * 2 + 1] = sender.get_num_connections(0) if hasattr(sender, "get_num_connections") else -1
            except Exception:
                stats[index * 2 + 1] = -1
    try:
        sender.close()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Publish realistic emulated NDI cameras from this PC.")
    ap.add_argument("--streams", type=int, default=8, help="number of emulated cameras (default 8)")
    ap.add_argument("--format", default="1080p30", choices=sorted(FORMATS), help="resolution/frame rate (default 1080p30)")
    ap.add_argument("--detail", type=float, default=5.5,
                    help="picture detail/noise 0-10; higher = higher NDI data rate (default 5.5 = ~110 Mbit/s at 1080p30; tune with the TX readout)")
    ap.add_argument("--name", default="VX3 EMU", help="source name prefix (sources appear as 'THIS-PC (VX3 EMU 01)')")
    ap.add_argument("--frames", type=int, default=0, help="distinct frames per camera loop (default: 1 s worth, 10 for 4K)")
    ap.add_argument("--status-seconds", type=float, default=2.0)
    args = ap.parse_args()

    w, h, fps = FORMATS[args.format]
    bank = args.frames or (10 if w > 1920 else fps)
    names = [f"{args.name} {i + 1:02d}" for i in range(args.streams)]
    try:
        import psutil
    except ImportError:
        psutil = None

    stop = mp.Event()
    stats = mp.RawArray("q", args.streams * 2)
    procs = [mp.Process(target=camera, args=(i, n, w, h, fps, args.detail, bank, stats, stop), daemon=True)
             for i, n in enumerate(names)]
    print(f"Starting {args.streams} emulated NDI camera(s): {args.format} ({w}x{h} @ {fps}), detail {args.detail}, "
          f"{bank} distinct frames each. Ctrl+C to stop.")
    for p in procs:
        p.start()

    # The signal handler only sets a plain flag. Setting the multiprocessing
    # Event from inside the handler while this thread is waiting on that same
    # Event deadlocks, so the main loop polls and signals the cameras itself.
    quit_requested = [False]

    def shutdown(*_):
        quit_requested[0] = True
    signal.signal(signal.SIGINT, shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, shutdown)

    last_t = time.perf_counter()
    last_sent = [0] * args.streams
    last_tx = psutil.net_io_counters().bytes_sent if psutil else None
    try:
        while not quit_requested[0]:
            end = time.perf_counter() + args.status_seconds
            while not quit_requested[0] and time.perf_counter() < end:
                time.sleep(0.1)
            if quit_requested[0]:
                break
            now = time.perf_counter()
            dt = now - last_t
            rates = []
            for i in range(args.streams):
                s = stats[i * 2]
                rates.append((s - last_sent[i]) / dt)
                last_sent[i] = s
            last_t = now
            line = "  ".join(f"{i + 1:02d}:{r:5.1f}fps/{stats[i * 2 + 1] if stats[i * 2 + 1] >= 0 else '?'}rx"
                             for i, r in enumerate(rates))
            tx = ""
            if psutil:
                cur = psutil.net_io_counters().bytes_sent
                mbps = (cur - last_tx) * 8 / dt / 1e6
                last_tx = cur
                receiving = sum(1 for i in range(args.streams) if stats[i * 2 + 1] > 0)
                per = f", {mbps / receiving:.0f} Mbit/s per received stream" if receiving else " (no receivers connected yet)"
                tx = f" | network TX {mbps:,.0f} Mbit/s{per}"
            dead = [n for n, p in zip(names, procs) if not p.is_alive()]
            print(f"[{time.strftime('%H:%M:%S')}] {line}{tx}" + (f" | STOPPED: {', '.join(dead)}" if dead else ""), flush=True)
            if dead and len(dead) == len(procs):
                print("All cameras stopped -- check that the NDI Runtime and cyndilib are installed.")
                break
    finally:
        print("Stopping cameras...", flush=True)
        stop.set()
        for p in procs:
            p.join(5)
            if p.is_alive():
                p.terminate()
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
