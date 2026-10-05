"""Engine: benchmarks NDI stream FPS, one dedicated process per stream.

Every stream runs in its own OS process for capture -- this is what
isolates "image receiving" so its throughput reflects the true,
unentangled capacity of the hardware and the NDI SDK, rather than being
contended for (via Python's GIL) with every other stream's capture work in
a single shared process.

Enabled AI models run as independent repeat-inference processes. They never
consume stream frames, so the stream measurement remains the receive rate
while system telemetry captures contention from both loads. With no models
enabled, no model process (and no multiprocessing manager) is started at all.
"""
import json,multiprocessing,sys,time
from datetime import datetime,timezone
from pathlib import Path
from .duration import duration_seconds
from .live_metrics import stream_snapshot_path, read_stream_history
from .live_publish import publish_json
from .stream_worker import run_stream_process
# NOTE: telemetry/report/inventory/sources/model/image-capture modules are
# imported inside run(), not here. On Windows every stream process is started
# with "spawn", which re-imports this module in each child; keeping the
# module top-level lean means 30 stream processes don't each load psutil,
# the report generator, etc. that they never use.


def _build_sources(cfg):
    """Synthetic test streams only. The benchmark never connects to NDI
    itself: NDI inputs are received by OBS (see _start_obs / obs_controller)."""
    sources = []
    for item in cfg["inputs"]:
        if not item.get("enabled", True) or item.get("transport") != "synthetic":
            continue
        count = max(1, int(item.get("stream_count", 1)))
        for i in range(count):
            sid = item["id"] if count == 1 else f'{item["id"]}-{i+1}'
            sources.append({"input_cfg": item, "source_cfg": item, "transport": "synthetic", "stream_id": sid})
    return sources


def _ndi_input(cfg):
    """The (single) NDI input, with any legacy top-level `obs_ndi` settings
    folded in. Several NDI inputs are merged: counts add up and the first
    one's OBS settings are used (OBS runs once and receives them all)."""
    items = [i for i in cfg.get("inputs", []) if i.get("enabled", True) and i.get("transport") == "ndi"]
    if not items:
        return None, 0
    legacy = {k: v for k, v in (cfg.get("obs_ndi") or {}).items() if k != "enabled"}
    item = {**legacy, **items[0]}
    return item, sum(max(1, int(i.get("stream_count", 1) or 1)) for i in items)

def _read_stream_snapshot(out, stream_id, transport):
    try:
        return json.loads(stream_snapshot_path(out, stream_id).read_text(encoding="utf-8"))
    except Exception:
        return {"stream_id": stream_id, "transport": transport, "state": "starting",
                "frames": 0, "dropped": 0, "instant_fps": 0.0}


def _live_stream_row(snapshot):
    """The live UI only needs current counters; the full FPS history lives
    in each stream's own .history.json file and is merged in for the final
    report only."""
    return {k: v for k, v in snapshot.items() if k != "fps_history"}


def run(path):
    from .inventory import collect
    from .report import write
    from .telemetry import Sampler
    from .model_workers import ModelWorkload
    from .image_capture import ImageCaptureWorkload
    from .aux_workloads import ImagePersistenceWorkload
    from .encode_workload import EncodeWorkload

    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    out = Path(cfg["_output"])
    session_t0 = time.time()
    stop = multiprocessing.Event()
    telemetry_cfg = cfg.get("telemetry", {}) or {}
    telemetry = Sampler(out, telemetry_cfg.get("sample_hz", 2), stop, telemetry_cfg)
    telemetry.start()
    base = Path(cfg.get("_scenario_dir", Path(path).parent))
    image_capture = ImageCaptureWorkload(cfg.get("image_capture", {}), cfg.get("_snapshot_url"))
    image_capture.start()

    # Streams are resolved first: the AI throughput target is
    # target_fps_per_stream x the number of AI-enabled input streams, so the
    # model scaler needs to know how many streams this session really has
    # (NDI sources are discovered at run time).
    sources = _build_sources(cfg)
    # NDI: launch OBS, let DistroAV discover the sources, and assign a
    # different NDI source to each of the requested OBS NDI inputs. Done
    # BEFORE the AI models start, so their per-stream target uses the real
    # number of streams.
    ndi_item, ndi_count = _ndi_input(cfg)
    obs = None
    if ndi_item is not None:
        from .obs_controller import ObsNdiController
        # When OBS is used, the synthetic streams are played INSIDE OBS too
        # (as Media Sources in the same scene), so every stream of the session
        # loads OBS the same way; NDI streams that cannot be found are filled
        # with synthetic ones up to stream_count.
        synthetic = [{"stream_id": s["stream_id"], "width": int(s["input_cfg"].get("width", 1920)),
                      "height": int(s["input_cfg"].get("height", 1080)), "framerate": int(s["input_cfg"].get("framerate", 30)),
                      "input_cfg": s["input_cfg"]} for s in sources]
        obs = ObsNdiController(ndi_count, ndi_item, out, session_t0, synthetic=synthetic)
        obs.start()
        if obs.started_ok:
            sources = obs.stream_list()          # everything now runs in the OBS scene
    obs_ids = set(obs.names) if obs and obs.started_ok else set()
    ai_streams = sum(1 for s in sources if s["input_cfg"].get("ai_enabled", True) is not False)
    model_workload = ModelWorkload(cfg.get("ai", {}), base, out, session_t0)
    from .model_workers import SCALER_VERSION
    print(f"[ai] scaler version {SCALER_VERSION}", flush=True)
    model_workload.start(ai_streams)
    image_encode = EncodeWorkload(cfg.get("encode_test"), out, session_t0)
    image_encode.start()
    persistence = ImagePersistenceWorkload(cfg.get("image_persistence"), [g.name for g in model_workload.groups], out, session_t0)
    persistence.start()

    def aux_snapshot():
        return {"updated_at": time.time(), "encode": image_encode.snapshot(), "persistence": persistence.snapshot(),
                "obs": obs.summary() if obs else None}
    publish_json(out / "live-aux.json", aux_snapshot())
    publish_json(out / "live-ai.json", {"updated_at": time.time(), "models": model_workload.snapshot()})
    print(f"[run] streams: {len(obs_ids)} in the OBS scene"
          + (f" ({obs.summary()['ndi_streams']} NDI + {obs.summary()['synthetic_streams']} synthetic)" if obs_ids else "")
          + f", {len(sources) - len(obs_ids)} synthetic run by the benchmark; "
          f"{ai_streams} AI-enabled; {len(model_workload.groups)} AI model(s)"
          + "".join(f"; {g.name} target {g.target:.1f} FPS" for g in model_workload.groups), flush=True)

    workers = []
    for s in sources:
        if s["stream_id"] in obs_ids or s["transport"] != "synthetic":
            continue                      # streams in the OBS scene are not run by the benchmark
        spec = {
            "stream_id": s["stream_id"],
            "transport": s["transport"],
            "input_cfg": s["input_cfg"],
            "source_cfg": s["source_cfg"],
            "out_dir": str(out),
            "session_t0": session_t0,
            "fps_history_interval_seconds": float(cfg.get("telemetry", {}).get("fps_history_interval_seconds", 1.0)),
        }
        workers.append(multiprocessing.Process(target=run_stream_process, args=(spec, stop), daemon=True))

    started = time.time()
    duration=duration_seconds(cfg.get("session", {}))
    for w in workers:
        w.start()
    next_tick = time.monotonic()
    try:
        while time.time() - started < duration:
            now = time.time()
            obs_rows = {r["stream_id"]: r for r in obs.tick()} if obs else {}
            snap = [obs_rows[s["stream_id"]] if s["stream_id"] in obs_rows else
                    _read_stream_snapshot(out, s["stream_id"], s["transport"]) for s in sources]
            publish_json(out / "live-streams.json", {"updated_at": now, "streams": [_live_stream_row(x) for x in snap]})
            models = model_workload.tick() if model_workload.groups else []
            if model_workload.workers:
                publish_json(out / "live-ai.json", {"updated_at": now, "models": models})
            persistence.tick(models)
            if image_encode.enabled or persistence.enabled or obs:
                publish_json(out / "live-aux.json", aux_snapshot())
            fps = " ".join(f"{x['stream_id']} = {x.get('instant_fps', 0):.1f}" for x in snap[:8])
            ai = " ".join(f"{m.get('name')} = {float(m.get('instant_fps') or 0):.2f}/{m.get('target_fps', 0):.0f} x{m.get('instance_count')}" for m in models)
            print(f"[run] elapsed = {now - started:.0f}s streams = {len(snap)} frames = {sum(x.get('frames',0) for x in snap)} "
                  f"dropped = {sum(x.get('dropped',0) for x in snap)} fps[{fps}]" + (f" ai_fps[{ai}]" if ai else ""), flush=True)
            # Absolute 1 s cadence so the loop's own work doesn't stretch it.
            next_tick = max(next_tick + 1.0, time.monotonic() - 1.0)
            time.sleep(max(0.0, next_tick - time.monotonic()))
    except KeyboardInterrupt:
        print("[run] operator stop",flush=True)

    stop.set()
    [w.join(20) for w in workers]
    obs_final = {r["stream_id"]: r for r in obs.snapshot(final=True)} if obs else {}
    obs_summary = obs.summary() if obs else None
    if obs:
        obs.stop()
    persistence.stop()
    image_encode.stop()
    model_workload.stop()
    persistence.join(10)
    image_encode.join(20)
    image_capture.stop()
    telemetry.join(3)


    streams = []
    for s in sources:
        if s["stream_id"] in obs_final:
            streams.append(obs_final[s["stream_id"]])
            continue
        row = _read_stream_snapshot(out, s["stream_id"], s["transport"])
        row["fps_history"] = read_stream_history(out, s["stream_id"])
        streams.append(row)
    elapsed=time.time() - started
    models = model_workload.snapshot(final=True)
    encode_final, persist_final = image_encode.snapshot(), persistence.snapshot()
    strip = lambda d: {k: v for k, v in d.items() if k != "fps_history"} if d else d
    enc_live = dict(encode_final, streams=[strip(x) for x in encode_final["streams"]]) if encode_final else None
    publish_json(out / "live-aux.json", {"updated_at": time.time(), "encode": enc_live,
                                         "persistence": [strip(x) for x in persist_final], "obs": obs_summary})
    publish_json(out / "live-ai.json", {"updated_at": time.time(), "models": model_workload.snapshot()})
    payload={"generated_at":datetime.now(timezone.utc).isoformat(),"run_summary":{"scenario":cfg["name"],"configured_duration_seconds":duration,"actual_duration_seconds":elapsed,"streams":len(streams),"total_frames":sum(x.get("frames",0) for x in streams),"total_dropped":sum(x.get("dropped",0) for x in streams)},"inventory":collect(),"config":{k:v for k,v in cfg.items() if not k.startswith("_")},"streams":streams,"ai_models":models,"encode_test":encode_final,"image_persistence":persist_final,"obs_ndi":obs_summary,"telemetry_summary":telemetry.summary(),"telemetry":telemetry.samples}
    write(out,payload)
    model_workload.close()
    image_capture.close()
    return 0

if __name__=="__main__":
    sys.exit(run(sys.argv[1]))
