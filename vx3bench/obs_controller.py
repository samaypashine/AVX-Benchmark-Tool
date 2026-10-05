"""Receive the NDI streams in OBS Studio instead of in the benchmark.

At the start of a session the benchmark:
  1. writes its own OBS scene collection ("VX3 Benchmark") that loads the
     obs/vx3_source_profiler.lua script (your other collections are not
     touched), and makes sure obs-websocket's server is enabled (the original
     obs-websocket config is backed up and restored at the end);
  2. launches OBS on that collection with a session-only websocket port and
     password on the command line (minimized, no shutdown-check dialog);
  3. over obs-websocket (v5, built into OBS 28+): creates the scene
     "VX3 NDI", adds one DistroAV "NDI Source" per discovered NDI stream,
     and tiles them in a grid (a multiview);
  4. every second reads the Source Profiler numbers the Lua script exports
     (per-source "async input" FPS = frames DistroAV received and handed to
     OBS, averaged by OBS over ~5 s) plus OBS's own stats, and turns them
     into the same per-stream metrics the rest of the benchmark uses;
  5. at the end closes OBS (unless told to keep it open).

Requirements on the benchmark machine: OBS Studio 30.1+ (Source Profiler),
DistroAV, NDI Runtime. No extra Python packages (the websocket client below
uses only the standard library).
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

SCENE_NAME = "VX3 NDI"
COLLECTION_NAME = "VX3 Benchmark"
LUA_SCRIPT = Path(__file__).resolve().parent / "obs" / "vx3_source_profiler.lua"
NDI_KIND = "ndi_source"
MEDIA_KIND = "ffmpeg_source"
SYNTHETIC_FORMATS = {"720p30": (1280, 720, 30), "1080p30": (1920, 1080, 30),
                     "1080p60": (1920, 1080, 60), "4k30": (3840, 2160, 30)}


# --------------------------------------------------------------------------- #
# Minimal RFC 6455 websocket client (text frames only), stdlib only
# --------------------------------------------------------------------------- #
class _WebSocket:
    def __init__(self, host: str, port: int, timeout: float = 5.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET / HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
                           f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
                           f"Sec-WebSocket-Protocol: obswebsocket.json\r\n\r\n").encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("websocket handshake: connection closed")
            buf += chunk
        head, self.buf = buf.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            raise ConnectionError(f"websocket handshake refused: {head.splitlines()[0]!r}")
        expect = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest())
        if expect not in head:
            raise ConnectionError("websocket handshake: bad Sec-WebSocket-Accept")

    def _read(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("websocket closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        header = bytes([0x80 | opcode])
        n = len(payload)
        if n < 126:
            header += bytes([0x80 | n])
        elif n < 65536:
            header += bytes([0x80 | 126]) + struct.pack("!H", n)
        else:
            header += bytes([0x80 | 127]) + struct.pack("!Q", n)
        mask = os.urandom(4)
        self.sock.sendall(header + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))

    def send(self, text: str) -> None:
        self._send_frame(0x1, text.encode("utf-8"))

    def recv(self) -> str:
        message = b""
        while True:
            b1, b2 = self._read(2)
            opcode, fin = b1 & 0x0F, b1 & 0x80
            n = b2 & 0x7F
            if n == 126:
                n = struct.unpack("!H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack("!Q", self._read(8))[0]
            mask = self._read(4) if b2 & 0x80 else None
            data = self._read(n)
            if mask:
                data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
            if opcode == 0x8:
                raise ConnectionError("websocket closed by OBS")
            if opcode == 0x9:
                self._send_frame(0xA, data)
                continue
            if opcode == 0xA:
                continue
            message += data
            if fin:
                return message.decode("utf-8")

    def close(self) -> None:
        try:
            self._send_frame(0x8, b"")
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass


class ObsWebSocketError(RuntimeError):
    pass


class ObsNotReady(Exception):
    """OBS accepted the request but is still starting up (code 207)."""


class ObsClient:
    """obs-websocket v5 request client."""

    def __init__(self, host: str, port: int, password: str, timeout: float = 5.0):
        self.ws = _WebSocket(host, port, timeout)
        hello = json.loads(self.ws.recv())
        if hello.get("op") != 0:
            raise ObsWebSocketError(f"unexpected first message from OBS: {hello}")
        d = hello["d"]
        identify = {"rpcVersion": 1, "eventSubscriptions": 0}
        auth = d.get("authentication")
        if auth:
            secret = base64.b64encode(hashlib.sha256((password + auth["salt"]).encode()).digest()).decode()
            identify["authentication"] = base64.b64encode(
                hashlib.sha256((secret + auth["challenge"]).encode()).digest()).decode()
        self.ws.send(json.dumps({"op": 1, "d": identify}))
        reply = json.loads(self.ws.recv())
        if reply.get("op") != 2:
            raise ObsWebSocketError(f"OBS rejected the connection: {reply}")
        self.obs_websocket_version = d.get("obsWebSocketVersion")

    NOT_READY = 207          # obs-websocket: "OBS is not ready to perform the request"
    ready_timeout = 90.0     # seconds to keep retrying while OBS is still starting up

    def request(self, request_type: str, data: dict | None = None, ok_codes=(100,)) -> dict:
        """Send one request. obs-websocket accepts connections before OBS
        has finished loading (plugins, scene collection); requests sent in
        that window fail with code 207 "not ready". That is temporary, so it
        is retried until OBS is ready (up to ready_timeout) -- only real
        errors are raised."""
        deadline = time.time() + self.ready_timeout
        while True:
            try:
                return self._request_once(request_type, data, ok_codes)
            except ObsNotReady:
                if time.time() >= deadline:
                    raise ObsWebSocketError(f"{request_type}: OBS was still not ready after {self.ready_timeout:.0f} s")
                time.sleep(0.5)

    def _request_once(self, request_type: str, data: dict | None, ok_codes) -> dict:
        rid = uuid.uuid4().hex
        msg = {"op": 6, "d": {"requestType": request_type, "requestId": rid}}
        if data is not None:
            msg["d"]["requestData"] = data
        self.ws.send(json.dumps(msg))
        while True:
            reply = json.loads(self.ws.recv())
            if reply.get("op") == 7 and reply["d"].get("requestId") == rid:
                status = reply["d"].get("requestStatus", {})
                if not status.get("result") and status.get("code") == self.NOT_READY:
                    raise ObsNotReady(request_type)
                if not status.get("result") and status.get("code") not in ok_codes:
                    raise ObsWebSocketError(f"{request_type} failed ({status.get('code')}): {status.get('comment', '')}")
                return reply["d"].get("responseData") or {}

    def close(self):
        self.ws.close()


# --------------------------------------------------------------------------- #
# OBS installation / configuration helpers
# --------------------------------------------------------------------------- #
FLATPAK_ID = "com.obsproject.Studio"


def _flatpak_obs() -> bool:
    """True if OBS is installed as a Flatpak (common on Linux)."""
    if not sys.platform.startswith("linux") or not shutil.which("flatpak"):
        return False
    try:
        return subprocess.run(["flatpak", "info", FLATPAK_ID], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def find_obs(configured: str = "") -> str | None:
    if configured and Path(configured).exists():
        return configured
    if sys.platform == "win32":
        for base in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramFiles(x86)", "")):
            for sub in ("64bit", "arm64"):                       # x64 and Windows-on-ARM builds
                p = Path(base) / "obs-studio" / "bin" / sub / "obs64.exe"
                if base and p.exists():
                    return str(p)
    elif sys.platform == "darwin":
        p = Path("/Applications/OBS.app/Contents/MacOS/OBS")
        if p.exists():
            return str(p)
    found = shutil.which("obs") or shutil.which("obs64")
    if found:
        return found
    if _flatpak_obs():
        return "flatpak:" + FLATPAK_ID
    return None


def obs_config_dir(configured: str = "", exe: str = "") -> Path:
    if configured:
        return Path(configured)
    if exe.startswith("flatpak:"):
        # Flatpak apps keep their config inside ~/.var/app/<id>/
        return Path.home() / ".var" / "app" / FLATPAK_ID / "config" / "obs-studio"
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "obs-studio"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "obs-studio"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "obs-studio"


def _pick_key(keys, preferred, *fragments):
    if preferred in keys:
        return preferred
    for k in keys:
        if all(f in k.lower() for f in fragments):
            return k
    return None


# --------------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------------- #
class ObsNdiController:
    def __init__(self, stream_count: int, cfg: dict[str, Any] | None, out_dir: Path, session_t0: float,
                 synthetic: list[dict[str, Any]] | None = None):
        """cfg is the NDI input from the scenario (inputs[] entry with
        transport "ndi"): stream_count, OBS settings and optional filter.
        synthetic: the scenario's synthetic streams, which are played inside
        OBS as well so every stream of the session is in the scene."""
        cfg = cfg or {}
        self.cfg = cfg
        self.requested = max(1, int(stream_count or 1))
        self.synthetic_requested = list(synthetic or [])
        self.names: list[str] = []          # stream ids in scene order, filled by start()
        self.meta: dict[str, dict] = {}     # stream id -> {"kind", "ndi_source", "format", "input_cfg"}
        self.cache_dir = Path(out_dir).parent / "_vx3_cache"
        self.out_dir = Path(out_dir)
        self.session_t0 = session_t0
        self.port = int(cfg.get("websocket_port", 4455))
        self.password = secrets.token_urlsafe(18)
        self.profiler_file = self.out_dir / "obs-source-profiler.json"
        self.process = None
        self.client = None
        self.error = ""
        self.notes: list[str] = []
        self.input_names: dict[str, str] = {}
        self.obs_stats: dict[str, Any] = {}
        self._ws_config_backup = None
        self._ws_config_path = None
        self._per_stream: dict[str, dict] = {}
        self._last_profiler: dict[str, Any] = {}
        self.started_ok = False
        self.ndi_seen_by_obs: list[str] = []

    # ---- setup ------------------------------------------------------------
    def _write_collection(self, cfg_dir: Path) -> None:
        scenes = cfg_dir / "basic" / "scenes"
        scenes.mkdir(parents=True, exist_ok=True)
        collection = {
            "name": COLLECTION_NAME, "current_scene": SCENE_NAME, "current_program_scene": SCENE_NAME,
            "scene_order": [], "sources": [], "groups": [], "transitions": [], "quick_transitions": [],
            "modules": {"scripts-tool": [{"path": str(LUA_SCRIPT).replace("\\", "/"),
                                          "settings": {"output_path": str(self.profiler_file).replace("\\", "/"),
                                                       "source_kind": f"{NDI_KIND},{MEDIA_KIND}"}}]},
        }
        (scenes / "VX3_Benchmark.json").write_text(json.dumps(collection, indent=2), encoding="utf-8")

    def _enable_websocket_server(self, cfg_dir: Path) -> None:
        path = cfg_dir / "plugin_config" / "obs-websocket" / "config.json"
        self._ws_config_path = path
        try:
            current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        except Exception:
            current = None
        if current is not None and current.get("server_enabled") is True:
            return
        self._ws_config_backup = path.read_bytes() if path.exists() else b""
        new = dict(current or {})
        new["server_enabled"] = True
        new.setdefault("server_port", self.port)
        new.setdefault("alerts_enabled", False)
        new.setdefault("auth_required", True)
        new.setdefault("server_password", secrets.token_urlsafe(18))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(new, indent=2), encoding="utf-8")
        self.notes.append("obs-websocket server was disabled; enabled for this session (original settings restored at the end)")

    def _restore_websocket_config(self) -> None:
        if self._ws_config_backup is None or self._ws_config_path is None:
            return
        try:
            if self._ws_config_backup == b"":
                self._ws_config_path.unlink(missing_ok=True)
            else:
                self._ws_config_path.write_bytes(self._ws_config_backup)
        except Exception as exc:
            self.notes.append(f"could not restore obs-websocket config: {exc}")
        self._ws_config_backup = None

    def _launch(self, exe: str) -> None:
        if exe.startswith("flatpak:"):
            cmd, cwd = ["flatpak", "run", exe.split(":", 1)[1]], None
        else:
            cmd, cwd = [exe], str(Path(exe).parent)
        args = cmd + ["--collection", COLLECTION_NAME, "--disable-shutdown-check", "--multi",
                      "--websocket_port", str(self.port), "--websocket_password", self.password]
        if sys.platform in ("win32", "darwin"):
            args.append("--disable-updater")                  # the updater exists only on Windows/macOS
        if self.cfg.get("minimize", True):
            args.append("--minimize-to-tray")
        flags = 0x00000200 if os.name == "nt" else 0          # CREATE_NEW_PROCESS_GROUP (graceful close)
        # OBS's console output goes to a file in the session folder, so a
        # start-up failure can be explained instead of just "exited".
        self.obs_output = self.out_dir / "obs-output.log"
        self._obs_output_fh = open(self.obs_output, "wb")
        self.process = subprocess.Popen(args, cwd=cwd, stdout=self._obs_output_fh, stderr=subprocess.STDOUT,
                                        creationflags=flags)

    def _startup_failure(self, code: int) -> str:
        """Explain why OBS exited during start-up, from its console output
        and its own newest log file."""
        sig = {-6: "aborted (SIGABRT)", -11: "crashed (SIGSEGV)", -9: "was killed (SIGKILL)"}.get(code, f"exited with code {code}")
        msg = [f"OBS {sig} during startup"]
        interesting = ("error", "fatal", "failed", "could not", "cannot", "qt.qpa", "abort", "display", "wayland", "xcb", "segmentation")
        lines = []
        try:
            self._obs_output_fh.flush()
            text = self.obs_output.read_text(encoding="utf-8", errors="replace").splitlines()
            lines += [l.strip() for l in text if any(k in l.lower() for k in interesting)]
        except Exception:
            pass
        try:
            logs = sorted((self._cfg_dir / "logs").glob("*.txt"), key=lambda p: p.stat().st_mtime)
            if logs:
                text = logs[-1].read_text(encoding="utf-8", errors="replace").splitlines()
                lines += [l.strip() for l in text[-200:] if any(k in l.lower() for k in interesting)]
                msg.append(f"OBS log: {logs[-1]}")
        except Exception:
            pass
        if lines:
            msg.append("last messages: " + " | ".join(dict.fromkeys(lines[-6:])))
        if sys.platform.startswith("linux") and code == -6:
            msg.append("on Linux this is usually OBS being unable to open a display or its graphics (OpenGL) context: "
                       "run the benchmark from the desktop session (not SSH/service), or check the OBS log above")
        msg.append(f"full OBS console output: {self.obs_output}")
        return "; ".join(msg)

    def _connect(self, timeout: float) -> None:
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(self._startup_failure(self.process.returncode))
            try:
                self.client = ObsClient("127.0.0.1", self.port, self.password)
                self.client.ready_timeout = float(self.cfg.get("obs_ready_timeout_seconds", 90))
                # Wait here until OBS has finished loading (retries 207).
                t0 = time.time()
                self.client.request("GetVersion")
                self.client.request("GetSceneList")
                waited = time.time() - t0
                if waited > 1:
                    self.notes.append(f"OBS needed {waited:.0f} s after its websocket opened to finish loading")
                return
            except Exception as exc:
                last = exc
                time.sleep(1.0)
        raise RuntimeError(f"could not connect to obs-websocket on port {self.port}: {last}")

    def _ndi_sources_seen(self, input_name: str, name_key: str) -> list[str]:
        """The NDI sources DistroAV currently lists in the source dropdown
        of `input_name` -- OBS does the discovery, not the benchmark."""
        try:
            items = self.client.request("GetInputPropertiesListPropertyItems",
                                        {"inputName": input_name, "propertyName": name_key}).get("propertyItems", [])
        except ObsWebSocketError as exc:
            raise RuntimeError(f"'{name_key}' is not DistroAV's source-name setting in this OBS ({exc}); "
                               "set source_name_key on the NDI input") from exc
        return [str(it.get("itemValue") or it.get("itemName") or "") for it in items
                if str(it.get("itemValue") or it.get("itemName") or "")]

    def _eligible(self, names: list[str]) -> list[str]:
        flt = str(self.cfg.get("source_filter", "") or "").lower()
        out = []
        for n in names:
            low = n.lower()
            # Never receive OBS's own NDI outputs (DistroAV "Main/Preview
            # Output" of this machine) -- that would be a feedback loop.
            if low.endswith("(obs)") or low.endswith("(obs preview)"):
                continue
            if flt and flt not in low:
                continue
            out.append(n)
        return sorted(dict.fromkeys(out))

    def _synthetic_clip(self, width: int, height: int, fps: int) -> str:
        """A looping test clip in SpeedHQ (SHQ2, 4:2:2) -- the codec
        full-bandwidth NDI uses -- with moving content and noise tuned to
        ~100 Mbit/s at 1080p30 (a real NDI camera is ~100-125), so OBS spends
        about the same decode work per frame as on a real NDI source.
        Cached, so it is generated only once per format."""
        from .encode_workload import ffmpeg_candidates
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        clip = self.cache_dir / f"vx3_synthetic_shq2_{width}x{height}p{fps}.mov"
        if clip.exists() and clip.stat().st_size > 0:
            return str(clip)
        ffmpegs = ffmpeg_candidates()
        if not ffmpegs:
            raise RuntimeError("FFmpeg not found (needed once to create the synthetic stream clip)")
        tmp = clip.with_suffix(".tmp.mov")
        cmd = [ffmpegs[0], "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
               "-i", f"testsrc2=size={width}x{height}:rate={fps}", "-vf", "noise=alls=5:allf=t",
               "-t", "10", "-c:v", "speedhq", "-q:v", "2", "-pix_fmt", "yuv422p", str(tmp)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                           creationflags=0x08000000 if os.name == "nt" else 0)
        if r.returncode != 0 or not tmp.exists():
            raise RuntimeError(f"could not create synthetic clip: {(r.stderr or '').strip()[:200]}")
        tmp.replace(clip)
        return str(clip)

    def _synthetic_plan(self) -> list[dict[str, Any]]:
        """Synthetic streams to place in the scene: the scenario's synthetic
        inputs, plus fill for any NDI streams that could not be found."""
        plan = [dict(x) for x in self.synthetic_requested]
        shortfall = self.requested - len(self.names)
        if shortfall > 0 and self.cfg.get("fill_with_synthetic", True):
            fmt = str(self.cfg.get("synthetic_format", "1080p30")).lower()
            w, h, r = SYNTHETIC_FORMATS.get(fmt, SYNTHETIC_FORMATS["1080p30"])
            for k in range(shortfall):
                plan.append({"stream_id": f"SYN-FILL-{k + 1:02d} ({fmt})", "width": w, "height": h,
                             "framerate": r, "input_cfg": self.cfg, "fill": True})
            self.notes.append(f"{len(self.names)} of {self.requested} NDI streams found; "
                              f"{shortfall} synthetic {fmt} stream(s) added to reach {self.requested}")
        return plan

    def _build_scene(self) -> None:
        c = self.client
        kinds = c.request("GetInputKindList", {"unversioned": True}).get("inputKinds", [])
        if NDI_KIND not in kinds:
            raise RuntimeError("OBS has no 'NDI Source' input -- install DistroAV (and the NDI Runtime) in this OBS")
        defaults = c.request("GetInputDefaultSettings", {"inputKind": NDI_KIND}).get("defaultInputSettings", {})
        keys = list(defaults)
        # OBS only reports settings that HAVE a default value, and DistroAV
        # gives the source name (and some booleans) no default, so DistroAV's
        # own setting names are used; the source-name key is verified by
        # asking OBS for that property's dropdown items.
        name_key = str(self.cfg.get("source_name_key") or "ndi_source_name")
        extra = {}
        bw_key = _pick_key(keys, "ndi_bw_mode", "bw") or "ndi_bw_mode"
        extra[bw_key] = 1 if str(self.cfg.get("bandwidth", "highest")).lower() == "lowest" else 0   # 0 = highest
        extra["ndi_framesync"] = False                            # every received frame, not frame-sync
        extra["ndi_recv_hw_accel"] = bool(self.cfg.get("hardware_acceleration", True))
        audio_key = _pick_key(keys, "ndi_audio", "audio")
        if audio_key and isinstance(defaults.get(audio_key), bool) and not self.cfg.get("receive_audio", False):
            extra[audio_key] = False
        self.notes.append(f"DistroAV settings: {name_key}=<source>, " + ", ".join(f"{k}={v}" for k, v in extra.items()))

        c.request("CreateScene", {"sceneName": SCENE_NAME}, ok_codes=(100, 601))      # 601 = already exists
        c.request("SetCurrentProgramScene", {"sceneName": SCENE_NAME})

        # 1. One input with no source yet: its dropdown is DistroAV's list of
        #    NDI sources on the network. Wait for discovery to find enough.
        probe = "VX3 NDI 01"
        try:
            c.request("CreateInput", {"sceneName": SCENE_NAME, "inputName": probe, "inputKind": NDI_KIND,
                                      "inputSettings": dict(extra), "sceneItemEnabled": True})
        except ObsWebSocketError as exc:
            if "601" not in str(exc):
                raise
        timeout = float(self.cfg.get("discovery_timeout_seconds", 15))
        deadline = time.time() + timeout
        seen: list[str] = []
        while True:
            seen = self._eligible(self._ndi_sources_seen(probe, name_key))
            if len(seen) >= self.requested or time.time() >= deadline:
                break
            time.sleep(1.0)
        self.ndi_seen_by_obs = seen
        if not seen and not self.cfg.get("fill_with_synthetic", True):
            raise RuntimeError(f"OBS/DistroAV found no NDI sources within {timeout:.0f} s"
                               + (f" matching filter '{self.cfg.get('source_filter')}'" if self.cfg.get("source_filter") else ""))
        # 2. Each input gets a DIFFERENT source.
        self.names = seen[:self.requested]
        for n in self.names:
            self.meta[n] = {"kind": "ndi", "ndi_source": n, "input_cfg": self.cfg}
        if len(self.names) < self.requested and not self.cfg.get("fill_with_synthetic", True):
            self.notes.append(f"only {len(self.names)} NDI source(s) available for the {self.requested} requested")
        ndi_count = len(self.names)
        if ndi_count == 0:
            c.request("RemoveInput", {"inputName": probe}, ok_codes=(100, 600))   # unused, no NDI found
        synthetic = self._synthetic_plan()
        clips = {}
        for sp in synthetic:
            key = (int(sp["width"]), int(sp["height"]), int(sp["framerate"]))
            if key not in clips:
                clips[key] = self._synthetic_clip(*key)
            sid = sp["stream_id"]
            self.names.append(sid)
            self.meta[sid] = {"kind": "synthetic", "clip": clips[key], "format": f"{key[0]}x{key[1]}p{key[2]}",
                              "input_cfg": sp.get("input_cfg") or self.cfg, "fill": bool(sp.get("fill"))}
        for n in self.names:
            self._per_stream[n] = {"history": [], "sum": 0.0, "count": 0, "min": None, "max": None, "frames": 0.0,
                                   "last_t": None, "first_t": None, "last": {}}

        video = c.request("GetVideoSettings")
        cw, ch = int(video.get("baseWidth", 1920)), int(video.get("baseHeight", 1080))
        n = len(self.names)
        cols = 1
        while cols * cols < n:
            cols += 1
        rows = -(-n // cols)
        tw, th = cw / cols, ch / rows
        media_settings = {"is_local_file": True, "looping": True, "restart_on_activate": False,
                          "close_when_inactive": False, "clear_on_media_end": False, "hw_decode": False,
                          "speed_percent": 100}
        syn_no = 0
        for i, name in enumerate(self.names):
            if self.meta[name]["kind"] == "synthetic":
                syn_no += 1
                input_name = f"VX3 SYN {syn_no:02d}"
                settings = {**media_settings, "local_file": self.meta[name]["clip"]}
                try:
                    item_id = c.request("CreateInput", {"sceneName": SCENE_NAME, "inputName": input_name, "inputKind": MEDIA_KIND,
                                                        "inputSettings": settings, "sceneItemEnabled": True}).get("sceneItemId")
                except ObsWebSocketError as exc:
                    if "601" not in str(exc):
                        raise
                    c.request("SetInputSettings", {"inputName": input_name, "inputSettings": settings, "overlay": True})
                    item_id = c.request("GetSceneItemId", {"sceneName": SCENE_NAME, "sourceName": input_name}).get("sceneItemId")
                self.input_names[name] = input_name
                if item_id is not None and self.cfg.get("arrange_grid", True):
                    col, row = i % cols, i // cols
                    c.request("SetSceneItemTransform", {"sceneName": SCENE_NAME, "sceneItemId": item_id, "sceneItemTransform": {
                        "positionX": col * tw, "positionY": row * th, "boundsType": "OBS_BOUNDS_SCALE_INNER",
                        "boundsWidth": tw, "boundsHeight": th, "boundsAlignment": 0, "alignment": 5}})
                continue
            input_name = f"VX3 NDI {i + 1:02d}"
            settings = {name_key: name, **extra}
            if i == 0:
                c.request("SetInputSettings", {"inputName": input_name, "inputSettings": settings, "overlay": True})
                item_id = c.request("GetSceneItemId", {"sceneName": SCENE_NAME, "sourceName": input_name}).get("sceneItemId")
            else:
                try:
                    item_id = c.request("CreateInput", {"sceneName": SCENE_NAME, "inputName": input_name, "inputKind": NDI_KIND,
                                                        "inputSettings": settings, "sceneItemEnabled": True}).get("sceneItemId")
                except ObsWebSocketError as exc:
                    if "601" not in str(exc):
                        raise
                    c.request("SetInputSettings", {"inputName": input_name, "inputSettings": settings, "overlay": True})
                    item_id = c.request("GetSceneItemId", {"sceneName": SCENE_NAME, "sourceName": input_name}).get("sceneItemId")
            self.input_names[name] = input_name
            if item_id is not None and self.cfg.get("arrange_grid", True):
                col, row = i % cols, i // cols
                c.request("SetSceneItemTransform", {"sceneName": SCENE_NAME, "sceneItemId": item_id, "sceneItemTransform": {
                    "positionX": col * tw, "positionY": row * th, "boundsType": "OBS_BOUNDS_SCALE_INNER",
                    "boundsWidth": tw, "boundsHeight": th, "boundsAlignment": 0, "alignment": 5}})

    def start(self, connect_timeout: float = 60.0) -> None:
        exe = find_obs(str(self.cfg.get("obs_path", "") or ""))
        if not exe:
            self.error = ("OBS Studio not found: set the NDI input's OBS executable (obs_path)" + (" to obs64.exe" if os.name == "nt" else "; on Linux install OBS (distribution package, PPA or Flatpak com.obsproject.Studio)"))
            print(f"[obs] {self.error}", flush=True)
            return
        cfg_dir = obs_config_dir(str(self.cfg.get("config_dir", "") or ""), exe)
        self._cfg_dir = cfg_dir
        if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            self.error = ("no graphical display: OBS is a desktop application and needs one. Start the benchmark "
                          "from a desktop session (not SSH or a system service), or export DISPLAY (e.g. DISPLAY=:0), "
                          "or run it under a virtual display: xvfb-run -a python -m vx3bench")
            print(f"[obs] {self.error}", flush=True)
            return
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.profiler_file.unlink(missing_ok=True)
            self._write_collection(cfg_dir)
            self._enable_websocket_server(cfg_dir)
            print(f"[obs] launching {exe} (collection '{COLLECTION_NAME}', websocket port {self.port})", flush=True)
            self._launch(exe)
            self._connect(connect_timeout)
            self._build_scene()
            self.started_ok = True
            nd = sum(1 for m in self.meta.values() if m.get("kind") == "ndi")
            print(f"[obs] scene '{SCENE_NAME}' ready: {len(self.input_names)} stream(s) ({nd} NDI + "
                  f"{len(self.input_names) - nd} synthetic) -> "
                  + ", ".join(f"{v}={k}" for k, v in self.input_names.items()) + "; " + "; ".join(self.notes), flush=True)
        except Exception as exc:
            self.error = str(exc)
            print(f"[obs] setup failed: {exc}", flush=True)

    # ---- per-second update -------------------------------------------------
    def _read_profiler(self) -> dict[str, Any]:
        try:
            data = json.loads(self.profiler_file.read_text(encoding="utf-8"))
            self._last_profiler = data
        except Exception:
            data = self._last_profiler
        return data

    def tick(self) -> list[dict[str, Any]]:
        now = time.time()
        prof = self._read_profiler()
        if self.client is not None:
            try:
                self.obs_stats = self.client.request("GetStats")
            except Exception as exc:
                self.obs_stats = {"error": str(exc)}
        sources = prof.get("sources", {}) if isinstance(prof, dict) else {}
        for name in self.names:
            st = self._per_stream[name]
            src = sources.get(self.input_names.get(name, ""), {})
            st["last"] = src
            fps = src.get("async_input_fps") if src.get("profiled") else None
            if fps is None:
                st["last_t"] = now
                continue
            if st["last_t"] is not None and fps > 0:
                st["frames"] += fps * (now - st["last_t"])          # estimate: profiler gives rates, not counts
            st["last_t"] = now
            if fps > 0 and st["first_t"] is None:
                st["first_t"] = now
            if st["first_t"] is not None:
                st["history"].append({"elapsed_seconds": round(now - self.session_t0, 3), "fps": fps,
                                      "fps_min": fps, "fps_max": fps, "interval_seconds": 1.0,
                                      "rendered_fps": src.get("async_rendered_fps")})
                st["sum"] += fps
                st["count"] += 1
                st["min"] = fps if st["min"] is None else min(st["min"], fps)
                st["max"] = fps if st["max"] is None else max(st["max"], fps)
        return self.snapshot()

    def _row(self, name: str, final: bool) -> dict[str, Any]:
        st = self._per_stream[name]
        src = st["last"]
        state = ("failed" if self.error else "running" if src.get("profiled") and (src.get("async_input_fps") or 0) > 0
                 else "connecting" if self.started_ok else "starting")
        if final and state == "running":
            state = "stopped"
        mean = st["sum"] / st["count"] if st["count"] else None
        meta = self.meta.get(name, {})
        synthetic = meta.get("kind") == "synthetic"
        # Sources published by ndi_camera_emulator.py (on a second PC) are
        # real NDI over the network, but not real cameras: label them.
        emulated = (not synthetic) and f"({str(self.cfg.get('emulator_name', 'VX3 EMU')).lower()}" in name.lower()
        row = {
            "stream_id": name, "transport": "synthetic" if synthetic else "ndi", "state": state,
            "ndi_receive_mode": (f"OBS Media Source playing a synthetic SpeedHQ {meta.get('format')} clip"
                                 + (" (fills a missing NDI stream)" if meta.get("fill") else "")
                                 + "; FPS from OBS Source Profiler, ~5 s average") if synthetic else
                                ("OBS + DistroAV" + (", emulated camera (ndi_camera_emulator.py on another PC)" if emulated else "")
                                 + " (FPS from OBS Source Profiler: async input, ~5 s average; frame counts estimated)"),
            "emulated_camera": emulated,
            "synthetic_fill": bool(meta.get("fill")),
            "obs_input_name": self.input_names.get(name, ""),
            "instant_fps": src.get("async_input_fps") or 0.0, "fps": mean, "fps_mean": mean,
            "fps_min": st["min"], "fps_max": st["max"], "frames": int(st["frames"]), "frames_estimated": True,
            "dropped": 0, "width": src.get("width"), "height": src.get("height"),
            "obs_rendered_fps": src.get("async_rendered_fps"), "obs_render_avg_ms": src.get("render_avg_ms"),
            "obs_render_max_ms": src.get("render_max_ms"), "obs_tick_avg_ms": src.get("tick_avg_ms"),
            "obs_async_input_worst_ms": src.get("async_input_worst_ms"),
            "active_seconds": (time.time() - st["first_t"]) if st["first_t"] else 0.0,
            "last_error": self.error or ("" if src.get("profiled") or not self._last_profiler.get("profiler_error")
                                         else self._last_profiler.get("profiler_error")),
        }
        if final:
            row["fps_history"] = st["history"]
        return row

    def snapshot(self, final: bool = False) -> list[dict[str, Any]]:
        return [self._row(n, final) for n in self.names]

    def stream_list(self) -> list[dict[str, Any]]:
        """Every stream in the OBS scene, for the engine's stream list."""
        return [{"stream_id": n, "transport": self.meta[n]["kind"], "input_cfg": self.meta[n].get("input_cfg") or self.cfg,
                 "source_cfg": None} for n in self.names]

    def summary(self) -> dict[str, Any]:
        s = self.obs_stats or {}
        return {"enabled": True, "started": self.started_ok, "error": self.error, "notes": self.notes,
                "websocket_port": self.port, "scene": SCENE_NAME, "sources": len(self.input_names),
                "requested_sources": self.requested,
                "assignments": [{"obs_input": v, "ndi_source": k if self.meta.get(k, {}).get("kind") == "ndi" else None,
                                 "synthetic": self.meta.get(k, {}).get("format") if self.meta.get(k, {}).get("kind") == "synthetic" else None,
                                 "stream_id": k} for k, v in self.input_names.items()],
                "ndi_streams": sum(1 for m in self.meta.values() if m.get("kind") == "ndi"),
                "synthetic_streams": sum(1 for m in self.meta.values() if m.get("kind") == "synthetic"),
                "profiler_available": self._last_profiler.get("profiler_available"),
                "ndi_sources_seen_by_obs": self.ndi_seen_by_obs,
                "profiler_error": self._last_profiler.get("profiler_error", ""),
                "obs_version_stats": {k: s.get(k) for k in ("activeFps", "cpuUsage", "memoryUsage", "averageFrameRenderTime",
                                                            "renderSkippedFrames", "renderTotalFrames",
                                                            "outputSkippedFrames", "outputTotalFrames") if k in s},
                "script_obs": self._last_profiler.get("obs", {})}

    # ---- shutdown ----------------------------------------------------------
    def stop(self) -> None:
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None
        if self.process is not None and not self.cfg.get("keep_obs_open", False):
            try:
                if os.name == "nt":
                    # WM_CLOSE to OBS's windows = a normal exit (saves config).
                    subprocess.run(["taskkill", "/PID", str(self.process.pid)], capture_output=True, timeout=10)
                else:
                    self.process.send_signal(signal.SIGTERM)
                self.process.wait(20)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        self._restore_websocket_config()
        fh = getattr(self, "_obs_output_fh", None)
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
