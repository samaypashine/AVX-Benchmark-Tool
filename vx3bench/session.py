import json,os,signal,subprocess,sys,threading
from pathlib import Path
from datetime import datetime,timezone
from .config import slug
from .duration import duration_seconds
class Manager:
    def __init__(self,root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents = True, exist_ok = True)
        self.process = None
        self.record = None

    def status(self):
        if not self.record:
            return {"status":"idle","telemetry":{"samples":[]},"stream_metrics":{"streams":[]}}
        
        data = dict(self.record)
        out = Path(data["output_dir"])
        log = out/"session.log"
        data["log_tail"] = "\n".join(log.read_text(encoding = "utf-8", errors = "replace").splitlines()[-500:]) if log.exists() else ""
        for key,name,default in [("telemetry","live-telemetry.json",{"samples":[]}),("stream_metrics","live-streams.json",{"streams":[]}),("ai_metrics","live-ai.json",{"models":[]}),("aux_metrics","live-aux.json",{"encode":None,"persistence":[]})]:
            try:
                data[key] = json.loads((out / name).read_text(encoding="utf-8"))
            except Exception:
                data[key]=default
        return data

    def start(self, scenario, snapshot_url=None):
        if self.process and self.process.poll() is None:
            raise RuntimeError("session active")
        sid = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + slug(scenario.name)
        out = self.root / slug(scenario.name) / sid
        out.mkdir(parents = True)
        cfg = dict(scenario.data)

        cfg["_output"] = str(out)
        cfg["_scenario_dir"] = str(scenario.path.parent.resolve())
        if snapshot_url:
            cfg["_snapshot_url"] = snapshot_url

        (out/"effective-config.json").write_text(json.dumps(cfg, indent=2),encoding="utf-8")
        (out/"scenario.json").write_text(json.dumps(scenario.data, indent=2),encoding="utf-8")
        self.record = {
            "id":sid,
            "scenario":scenario.name,
            "status":"running",
            "configured_duration_seconds":duration_seconds(cfg.get("session",{})),
            "started_at":datetime.now(timezone.utc).isoformat(),
            "finished_at":None,
            "output_dir":str(out),
            "report_path":None
            }

        self.log = (out / "session.log").open("w", encoding="utf-8", buffering = 1)
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.process = subprocess.Popen([sys.executable,
                                         "-m","vx3bench.engine",
                                         str(out / "effective-config.json")],
                                         stdout=self.log,
                                         stderr=subprocess.STDOUT,
                                         creationflags=flags,
                                         cwd=str(Path(__file__).parents[1]))
        threading.Thread(target = self._watch, args = (out, ), daemon = True).start()
        return self.record

    def stop(self):
        if not self.process or self.process.poll() is not None:
            raise RuntimeError("no active session")
        self.record["status"] = "stopping"
        self.process.send_signal(signal.CTRL_BREAK_EVENT if os.name=="nt" else signal.SIGINT)
        return self.record

    def _watch(self, out):
        code = self.process.wait()
        self.log.close()
        self.record["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.record["report_path"] = str(out/"report.html") if (out/"report.html").exists() else None
        self.record["report_pdf_path"] = str(out/"report.pdf") if (out/"report.pdf").exists() else None
        self.record["status"] = "completed" if code == 0 else "failed"
        self.record["exit_code"] = code
        (out / "manifest.json").write_text(json.dumps(self.record,indent=2), encoding="utf-8")
        self.process = None
