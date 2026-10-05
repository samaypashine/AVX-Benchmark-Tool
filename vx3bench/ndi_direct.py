"""Direct NDI receive -- the same SDK call OBS (DistroAV) uses by default.

Each stream thread blocks in NDIlib_recv_capture_v3(), which returns every
video frame exactly ONCE, when it actually arrives -- never a repeat of an
old frame, never a frame fabricated to fill a clock tick. That makes every
returned frame a genuinely unique frame, so counting them is the true
received frame rate.

On top of that, every NDI frame carries the SENDER's timestamp (100 ns
units). Consecutive timestamps reveal what happened on the way:
  * gap of ~1 frame interval  -> normal;
  * gap of k intervals (k > 1.5) -> k-1 frames the sender produced never
    reached us (network loss or the SDK's queue overflowing because this
    machine was too slow) -> counted as `missing_frames`;
  * same or older timestamp   -> a repeated/out-of-order frame -> counted
    as `duplicate_frames` and NOT counted as a received frame.

The SDK is called through ctypes (which releases Python's GIL while it
waits), using the documented NDI C structures. The library is found via the
NDI Runtime environment variables, next to cyndilib, or on the system path.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
import sys
import threading
import time
from pathlib import Path

FRAME_NONE, FRAME_VIDEO, FRAME_AUDIO, FRAME_METADATA, FRAME_ERROR, FRAME_STATUS = 0, 1, 2, 3, 4, 100
COLOR = {"bgrx": 0, "uyvy": 1, "rgbx": 2}          # NDIlib_recv_color_format_*
BANDWIDTH = {"lowest": 0, "highest": 100}
TIMESTAMP_UNDEFINED = 0x7FFFFFFFFFFFFFFF            # NDIlib_recv_timestamp_undefined
FOURCC_UYVY = 0x59565955


class NDIlib_source_t(ctypes.Structure):
    _fields_ = [("p_ndi_name", ctypes.c_char_p), ("p_url_address", ctypes.c_char_p)]


class NDIlib_recv_create_v3_t(ctypes.Structure):
    _fields_ = [("source_to_connect_to", NDIlib_source_t), ("color_format", ctypes.c_int),
                ("bandwidth", ctypes.c_int), ("allow_video_fields", ctypes.c_bool),
                ("p_ndi_recv_name", ctypes.c_char_p)]


class NDIlib_video_frame_v2_t(ctypes.Structure):
    _fields_ = [("xres", ctypes.c_int), ("yres", ctypes.c_int), ("FourCC", ctypes.c_int),
                ("frame_rate_N", ctypes.c_int), ("frame_rate_D", ctypes.c_int),
                ("picture_aspect_ratio", ctypes.c_float), ("frame_format_type", ctypes.c_int),
                ("timecode", ctypes.c_int64), ("p_data", ctypes.c_void_p),
                ("line_stride_in_bytes", ctypes.c_int), ("p_metadata", ctypes.c_char_p),
                ("timestamp", ctypes.c_int64)]


class NDIlib_metadata_frame_t(ctypes.Structure):
    _fields_ = [("length", ctypes.c_int), ("timecode", ctypes.c_int64), ("p_data", ctypes.c_char_p)]


_lib = None
_lib_lock = threading.Lock()
_lib_error = ""


def _candidates():
    names = (["Processing.NDI.Lib.x64.dll"] if sys.platform == "win32" else
             ["libndi.dylib"] if sys.platform == "darwin" else ["libndi.so.6", "libndi.so.5", "libndi.so"])
    paths = []
    override = os.environ.get("VX3_NDI_LIB")
    if override:
        paths.append(override)
    for var in ("NDI_RUNTIME_DIR_V6", "NDI_RUNTIME_DIR_V5", "NDI_RUNTIME_DIR_V4"):
        d = os.environ.get(var)
        if d:
            paths += [str(Path(d) / n) for n in names]
    try:  # cyndilib ships/uses the NDI library; look next to it
        import cyndilib
        base = Path(cyndilib.__file__).parent
        for n in names:
            paths += glob.glob(str(base / "**" / n), recursive=True)
    except Exception:
        pass
    found = ctypes.util.find_library("ndi")
    if found:
        paths.append(found)
    paths += names          # let the OS loader search its default paths
    seen = []
    for p in paths:
        if p not in seen:
            seen.append(p)
    return seen


def load_library():
    """Load and initialise the NDI SDK once per process."""
    global _lib, _lib_error
    with _lib_lock:
        if _lib is not None:
            return _lib
        errors = []
        for path in _candidates():
            try:
                if sys.platform == "win32" and os.path.isabs(path):
                    try:
                        os.add_dll_directory(str(Path(path).parent))
                    except Exception:
                        pass
                lib = ctypes.CDLL(path)
                break
            except OSError as exc:
                errors.append(f"{path}: {exc}")
        else:
            _lib_error = "NDI library not found (install the NDI Runtime or set VX3_NDI_LIB): " + "; ".join(errors[:3])
            raise OSError(_lib_error)
        lib.NDIlib_initialize.restype = ctypes.c_bool
        lib.NDIlib_recv_create_v3.argtypes = [ctypes.POINTER(NDIlib_recv_create_v3_t)]
        lib.NDIlib_recv_create_v3.restype = ctypes.c_void_p
        lib.NDIlib_recv_capture_v3.argtypes = [ctypes.c_void_p, ctypes.POINTER(NDIlib_video_frame_v2_t),
                                               ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        lib.NDIlib_recv_capture_v3.restype = ctypes.c_int
        lib.NDIlib_recv_free_video_v2.argtypes = [ctypes.c_void_p, ctypes.POINTER(NDIlib_video_frame_v2_t)]
        lib.NDIlib_recv_free_video_v2.restype = None
        lib.NDIlib_recv_destroy.argtypes = [ctypes.c_void_p]
        lib.NDIlib_recv_destroy.restype = None
        lib.NDIlib_recv_send_metadata.argtypes = [ctypes.c_void_p, ctypes.POINTER(NDIlib_metadata_frame_t)]
        lib.NDIlib_recv_send_metadata.restype = ctypes.c_bool
        lib.NDIlib_recv_get_no_connections.argtypes = [ctypes.c_void_p]
        lib.NDIlib_recv_get_no_connections.restype = ctypes.c_int
        if not lib.NDIlib_initialize():
            raise OSError("NDIlib_initialize() failed (CPU not supported by the NDI SDK?)")
        _lib = lib
        return lib


class DirectReceiver:
    """One NDI receiver, read with blocking recv_capture_v3 (OBS's method)."""

    def __init__(self, source_name: str, url: str | None = None, color_format: str = "uyvy",
                 bandwidth: str = "highest", hw_accel: bool = True, recv_name: str = "VX3 Benchmark"):
        self.lib = load_library()
        self._name = source_name.encode("utf-8")
        self._url = url.encode("utf-8") if url else None
        self._recv_name = recv_name.encode("utf-8")
        create = NDIlib_recv_create_v3_t(NDIlib_source_t(self._name, self._url),
                                         COLOR.get(color_format, 1), BANDWIDTH.get(bandwidth, 100),
                                         False, self._recv_name)
        self.handle = self.lib.NDIlib_recv_create_v3(ctypes.byref(create))
        if not self.handle:
            raise RuntimeError(f"NDIlib_recv_create_v3 failed for {source_name!r}")
        self.color_format = {0: "BGRX", 1: "UYVY", 2: "RGBX"}.get(COLOR.get(color_format, 1), "UYVY")
        self.hw_accel = self._send_hwaccel() if hw_accel else "off"
        self.frame = NDIlib_video_frame_v2_t()
        # Measurement state
        self.last_ts = None
        self.missing = 0
        self.duplicates = 0
        self.frame_rate = 0.0

    def _send_hwaccel(self) -> str:
        xml = b'<ndi_hwaccel enabled="true"/>'
        md = NDIlib_metadata_frame_t(len(xml) + 1, 0, xml)
        try:
            return "requested" if self.lib.NDIlib_recv_send_metadata(self.handle, ctypes.byref(md)) else "not accepted"
        except Exception as exc:
            return f"failed: {exc}"

    def connections(self) -> int:
        return int(self.lib.NDIlib_recv_get_no_connections(self.handle))

    def capture(self, timeout_ms: int = 500):
        """Block up to timeout_ms for the next genuinely new video frame.

        Returns (xres, yres, arrived_at) for a new frame, None on timeout,
        and raises on an SDK error frame. Audio/metadata are discarded by
        the SDK (NULL pointers), as for a video-only receiver."""
        kind = self.lib.NDIlib_recv_capture_v3(self.handle, ctypes.byref(self.frame), None, None, timeout_ms)
        arrived = time.time()
        if kind == FRAME_ERROR:
            raise RuntimeError("NDI receiver reported an error (connection lost)")
        if kind != FRAME_VIDEO:
            return None
        f = self.frame
        try:
            x, y = int(f.xres), int(f.yres)
            if f.frame_rate_N > 0 and f.frame_rate_D > 0:
                self.frame_rate = f.frame_rate_N / f.frame_rate_D
            ts = f.timestamp if f.timestamp not in (0, TIMESTAMP_UNDEFINED) else f.timecode
        finally:
            self.lib.NDIlib_recv_free_video_v2(self.handle, ctypes.byref(self.frame))
        if self.last_ts is not None and self.frame_rate > 0:
            interval = 1e7 / self.frame_rate          # 100 ns ticks per frame
            delta = ts - self.last_ts
            if delta <= 0:
                self.duplicates += 1                   # repeated / out-of-order frame
                return "duplicate"
            if delta > 1.5 * interval:
                self.missing += int(round(delta / interval)) - 1
        self.last_ts = ts
        return x, y, arrived

    def close(self):
        if getattr(self, "handle", None):
            self.lib.NDIlib_recv_destroy(self.handle)
            self.handle = None
