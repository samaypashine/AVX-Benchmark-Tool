import json, re
from dataclasses import dataclass
from pathlib import Path
from .duration import duration_seconds

@dataclass
class Scenario:
    path: Path
    data: dict
    @property
    def name(self): return self.data.get("name", self.path.stem)

def validate(d):
    if not isinstance(d, dict) or not str(d.get("name", "")).strip():
        raise ValueError("name is required")
    duration_seconds(d.get("session", {}))
    if not isinstance(d.get("inputs"), list) or not d["inputs"]:
        raise ValueError("inputs must be a non-empty array")
    for i, item in enumerate(d["inputs"]):
        if item.get("transport") not in ("ndi", "synthetic"):
            raise ValueError(f"inputs[{i}].transport must be ndi or synthetic")
    models = d.get("ai", {}).get("models", {})
    if not isinstance(models, dict):
        raise ValueError("ai.models must be an object")
    for section in ("image_persistence",):
        sec = d.get(section, {}) or {}
        if not isinstance(sec, dict):
            raise ValueError(f"{section} must be an object")
        for key in ("width", "height"):
            v = sec.get(key, 1920)
            if isinstance(v, bool) or not isinstance(v, int) or not (16 <= v <= 16384):
                raise ValueError(f"{section}.{key} must be an integer between 16 and 16384")
        if str(sec.get("format", "jpeg")).lower() not in ("jpeg", "png", "webp"):
            raise ValueError(f"{section}.format must be jpeg, png or webp")
        q = sec.get("quality", 90)
        if isinstance(q, bool) or not isinstance(q, int) or not (1 <= q <= 100):
            raise ValueError(f"{section}.quality must be an integer from 1 to 100")
        for key in ("enabled", "fsync"):
            if key in sec and not isinstance(sec[key], bool):
                raise ValueError(f"{section}.{key} must be true or false")
    enc = d.get("encode_test", {}) or {}
    if not isinstance(enc, dict):
        raise ValueError("encode_test must be an object")
    if str(enc.get("resolution", "1080p30")).lower() not in ("1080p30", "4k30"):
        raise ValueError("encode_test.resolution must be 1080p30 or 4k30")
    n = enc.get("stream_count", 1)
    if isinstance(n, bool) or not isinstance(n, int) or not (1 <= n <= 64):
        raise ValueError("encode_test.stream_count must be an integer from 1 to 64")
    if "enabled" in enc and not isinstance(enc["enabled"], bool):
        raise ValueError("encode_test.enabled must be true or false")
    ndi_items = [x for x in d.get("inputs") or [] if isinstance(x, dict) and x.get("transport") == "ndi"]
    for n, item in enumerate(d.get("inputs") or []):
        if not isinstance(item, dict) or item.get("transport") != "ndi":
            continue
        # NDI streams are received by OBS (DistroAV), launched by the benchmark.
        cnt = item.get("stream_count", 1)
        if isinstance(cnt, bool) or not isinstance(cnt, int) or not (1 <= cnt <= 64):
            raise ValueError(f"inputs[{n}].stream_count must be a whole number from 1 to 64 (NDI sources to receive in OBS)")
        if str(item.get("synthetic_format", "1080p30")).lower() not in ("720p30", "1080p30", "1080p60", "4k30"):
            raise ValueError(f"inputs[{n}].synthetic_format must be 720p30, 1080p30, 1080p60 or 4k30")
        for key in ("minimize", "arrange_grid", "hardware_acceleration", "keep_obs_open", "enabled", "fill_with_synthetic"):
            if key in item and not isinstance(item[key], bool):
                raise ValueError(f"inputs[{n}].{key} must be true or false")
        port = item.get("websocket_port", 4455)
        if isinstance(port, bool) or not isinstance(port, int) or not (1024 <= port <= 65535):
            raise ValueError(f"inputs[{n}].websocket_port must be a port number between 1024 and 65535")
        if str(item.get("bandwidth", "highest")).lower() not in ("highest", "lowest"):
            raise ValueError(f"inputs[{n}].bandwidth must be highest or lowest")
        dt = item.get("discovery_timeout_seconds", 15)
        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not (1 <= dt <= 300):
            raise ValueError(f"inputs[{n}].discovery_timeout_seconds must be between 1 and 300")
    ai = d.get("ai", {})
    modes = [("ai.worker_mode", ai.get("worker_mode", "shared")),
             ("encode_test.worker_mode", (d.get("encode_test") or {}).get("worker_mode", "shared")),
             ("image_persistence.worker_mode", (d.get("image_persistence") or {}).get("worker_mode", "shared"))]
    modes += [(f"ai.models.{n}.worker_mode", m.get("worker_mode")) for n, m in (ai.get("models") or {}).items()
              if isinstance(m, dict) and "worker_mode" in m]
    for key, value in modes:
        if str(value).lower() not in ("shared", "process"):
            raise ValueError(f"{key} must be 'shared' or 'process'")
    tfps = ai.get("target_fps_per_stream", 15)
    if isinstance(tfps, bool) or not isinstance(tfps, (int, float)) or tfps < 0:
        raise ValueError("ai.target_fps_per_stream must be a non-negative number (0 disables the target)")
    scale = ai.get("autoscale", {})
    if not isinstance(scale, dict):
        raise ValueError("ai.autoscale must be an object")
    for key in ("enabled",):
        if key in scale and not isinstance(scale[key], bool):
            raise ValueError(f"ai.autoscale.{key} must be true or false")
    for key in ("max_instances", "settle_seconds", "tolerance_percent", "min_gain_percent"):
        value = scale.get(key, 1)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or (key == "max_instances" and value < 1):
            raise ValueError(f"ai.autoscale.{key} must be a {'positive' if key == 'max_instances' else 'non-negative'} number")
    gain = scale.get("min_gain_percent", 5)
    if isinstance(gain, bool) or not isinstance(gain, (int, float)) or gain <= 0:
        raise ValueError("ai.autoscale.min_gain_percent must be greater than 0 (the minimum-gain check is always enabled)")
    for name, model in models.items():
        if not isinstance(model, dict):
            raise ValueError(f"ai.models.{name} must be an object")
        if model.get("enabled") and not model.get("path"):
            raise ValueError(f"enabled model {name} requires path")
        if str(model.get("device", "auto")).strip().lower() not in ("auto", "cpu", "cuda", "gpu", "tensorrt", "directml", "coreml"):
            raise ValueError(f"ai.models.{name}.device must be auto, cuda, tensorrt, directml, coreml or cpu")
        for key in ("warmup_iterations", "device_id", "instances"):
            value = model.get(key, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"ai.models.{name}.{key} must be a non-negative integer")
    image_capture = d.get("image_capture", {})
    if not isinstance(image_capture, dict):
        raise ValueError("image_capture must be an object")
    polling_rate = image_capture.get("snapshot_polling_rate", 1000)
    if isinstance(polling_rate, bool) or not isinstance(polling_rate, int) or polling_rate <= 0:
        raise ValueError("image_capture.snapshot_polling_rate must be a positive integer number of milliseconds")

def load(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate(data)
    return Scenario(path, data)

def save(path, data):
    validate(data)
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)

def discover(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    good, bad = [], []
    for path in sorted(folder.glob("*.json")):
        try:
            s = load(path)
            good.append({"id": path.name, "name": s.name, "description": s.data.get("description", "")})
        except Exception as exc:
            bad.append({"id": path.name, "error": str(exc)})
    return {"configs": good, "invalid": bad, "config_dir": str(folder.resolve())}

def slug(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "session"

def default_scenario(name):
    """A minimal, already-valid scenario used to seed a brand new config from
    the WebUI, so a user can go from "New" to a running session with only a
    couple of clicks."""
    return {
        "name": name,
        "description": "",
        "session": {"duration": {"value": 5, "unit": "minutes"}},
        "inputs": [
            {
                "id": "synthetic",
                "transport": "synthetic",
                "enabled": True,
                "stream_count": 1,
                "width": 1920,
                "height": 1080,
                "framerate": 30,
            },
            {
                # Received by OBS: the benchmark launches OBS, creates scene
                # "VX3 NDI" with stream_count DistroAV NDI Sources, each
                # assigned a different discovered NDI source, and reads each
                # stream's FPS from OBS's Source Profiler.
                "id": "ndi",
                "transport": "ndi",
                "enabled": False,
                "stream_count": 12,
                "source_filter": "",
                "fill_with_synthetic": True,
                "synthetic_format": "1080p30",
                "discovery_timeout_seconds": 15,
                "bandwidth": "highest",
                "hardware_acceleration": True,
                "obs_path": "",
                "websocket_port": 4455,
                "minimize": True,
                "arrange_grid": True,
                "keep_obs_open": False,
            }
        ],
        "ai": {
            "target_fps_per_stream": 15,
            "worker_mode": "shared",
            "autoscale": {"enabled": True, "max_instances": 16, "settle_seconds": 8, "tolerance_percent": 2,
                          "min_gain_percent": 5},
            "models": {},
        },
        "image_capture": {"enabled": False, "snapshot_polling_rate": 1000},
        "encode_test": {"enabled": False, "resolution": "1080p30", "stream_count": 1, "worker_mode": "shared"},
        "image_persistence": {"enabled": False, "directory": "", "width": 1920, "height": 1080,
                              "format": "jpeg", "quality": 90, "fsync": True, "worker_mode": "shared"},
        "telemetry": {"sample_hz": 2},
        "targets": {
            "stream_fps_min": 29,
            "cpu_percent_max": 90,
            "gpu_percent_max": 95,
            "vram_percent_max": 90,
            "ram_percent_max": 90,
            "temperature_c_max": 90,
            "dropped_frame_percent_max": 1,
            "process_rss_growth_mb_max": 256,
            "disk_free_gb_min": 20,
        },
    }
