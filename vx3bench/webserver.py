import json
import mimetypes
from email import policy
from email.parser import BytesParser
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import discover, load, save, validate, default_scenario, slug
from .session import Manager

# Hardware inventory is static; collect() launches several PowerShell/WMI
# queries on Windows, so it is refreshed rarely and NEVER while a benchmark
# session is running (it used to re-run every 30 s because the UI polls it).
INVENTORY_CACHE_SECONDS = 600


class App:
    def __init__(self, config_dir, reports_dir, snapshot_url):
        self.configs = config_dir.resolve()
        self.static = Path(__file__).parent / "web/static"
        self.snapshot = Path(__file__).parent / "snapshot.jpeg"
        self.snapshot_url = snapshot_url
        self.sessions = Manager(reports_dir)
        self._inventory_cache = None
        self._inventory_cache_at = 0

    def cp(self, name):
        path = (self.configs / Path(name).name).resolve()
        if self.configs not in path.parents or not path.is_file():
            raise ValueError("unknown config")
        return path

    def create_config(self, name, data=None):
        name = (name or "").strip()
        if not name:
            raise ValueError("name is required")
        filename = slug(name) + ".json"
        path = self.configs / filename
        if path.exists():
            raise ValueError(f"a config named '{filename}' already exists")
        self.configs.mkdir(parents=True, exist_ok=True)
        save(path, data or default_scenario(name))
        return {"id": filename}

    def inventory(self):
        now = time.time()
        running = self.sessions.process is not None and self.sessions.process.poll() is None
        stale = now - self._inventory_cache_at > INVENTORY_CACHE_SECONDS
        if self._inventory_cache is None or (stale and not running):
            from .inventory import collect
            self._inventory_cache = collect()
            self._inventory_cache_at = now
        return self._inventory_cache

    def models(self, relative=""):
        root = (Path(__file__).parents[1] / "models").resolve()
        target = (root / Path(relative)).resolve()
        if root != target and root not in target.parents:
            raise ValueError("model path escapes the models directory")
        if not target.exists() or not target.is_dir():
            raise ValueError("model directory not found")
        return [{"name": p.name, "path": str(p.relative_to(root)).replace("\\", "/"), "directory": p.is_dir()}
                for p in sorted(target.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
                if p.is_dir() or p.suffix.lower() == ".onnx"]

    def upload_model(self, handler):
        root = (Path(__file__).parents[1] / "models").resolve()
        content_type = handler.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            raise ValueError("model upload must use multipart/form-data")
        length = int(handler.headers.get("Content-Length", "0"))
        raw = handler.rfile.read(length)
        message = BytesParser(policy=policy.default).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + raw
        )
        item = next((part for part in message.walk() if part.get_param("name", header="content-disposition") == "model"), None)
        filename = item.get_filename() if item else None
        if item is None or not filename:
            raise ValueError("choose an ONNX model file")
        filename = Path(filename).name
        if Path(filename).suffix.lower() != ".onnx":
            raise ValueError("only .onnx model files are supported")
        destination = (root / filename).resolve()
        if root not in destination.parents:
            raise ValueError("invalid model filename")
        with destination.open("wb") as output:
            output.write(item.get_payload(decode=True) or b"")
        return {"name": filename, "path": f"../models/{filename}", "directory": False}

    def handler(self):
        app = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                print("[web] " + fmt % args)

            def js(self, value, status=200):
                body = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def body(self):
                return json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")

            def do_GET(self):
                path = urllib.parse.urlparse(self.path).path
                if path == "/api/configs":
                    return self.js(discover(app.configs))
                if path.startswith("/api/configs/"):
                    try:
                        scenario = load(app.cp(urllib.parse.unquote(path.split("/", 3)[3])))
                        return self.js({"id": scenario.path.name, "data": scenario.data})
                    except Exception as exc:
                        return self.js({"error": str(exc)}, 400)
                if path == "/api/session":
                    return self.js(app.sessions.status())
                if path == "/api/inventory":
                    return self.js(app.inventory())
                if path == "/api/models":
                    try:
                        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                        return self.js({"root": "models", "entries": app.models(query.get("path", [""])[0])})
                    except Exception as exc:
                        return self.js({"error": str(exc)}, 400)
                if path.startswith("/reports/"):
                    report = (app.sessions.root / urllib.parse.unquote(path[9:])).resolve()
                    if app.sessions.root not in report.parents:
                        return self.send_error(404)
                    return self.file(report)
                if path == "/snapshot.jpeg":
                    return self.file(app.snapshot)
                static_file = app.static / ("index.html" if path == "/" else path.lstrip("/"))
                return self.file(static_file) if static_file.is_file() else self.send_error(404)

            def do_POST(self):
                try:
                    path = urllib.parse.urlparse(self.path).path
                    if path == "/api/models/upload":
                        return self.js(app.upload_model(self), 201)
                    data = self.body()
                    if path == "/api/configs/validate":
                        validate(data["data"])
                        return self.js({"valid": True})
                    if path == "/api/configs":
                        return self.js(app.create_config(data.get("name"), data.get("data")), 201)
                    if path == "/api/session/start":
                        return self.js(app.sessions.start(load(app.cp(data["config"])), app.snapshot_url), 201)
                    if path == "/api/session/stop":
                        return self.js(app.sessions.stop())
                except Exception as exc:
                    return self.js({"error": str(exc)}, 400)

            def do_PUT(self):
                try:
                    path = urllib.parse.urlparse(self.path).path
                    save(app.cp(urllib.parse.unquote(path.split("/", 3)[3])), self.body()["data"])
                    return self.js({"saved": True})
                except Exception as exc:
                    return self.js({"error": str(exc)}, 400)

            def file(self, path):
                body = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler


def serve(host, port, config_dir, reports_dir):
    server = ThreadingHTTPServer(
        (host, port),
        App(config_dir, reports_dir, f"http://127.0.0.1:{port}/snapshot.jpeg").handler(),
    )
    print(f"VX3 UI: http://localhost:{port}")
    server.serve_forever()