"""Optional snapshot download worker for camera image capture benchmarks."""
from __future__ import annotations

import multiprocessing
from pathlib import Path
import time
from urllib.request import urlopen


def _capture_process(url: str, polling_rate_ms: int, destination: str, state, lock, stop_event) -> None:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    interval = polling_rate_ms / 1000.0
    while not stop_event.is_set():
        started = time.perf_counter()
        try:
            with urlopen(url, timeout=max(1.0, interval)) as response:
                target.write_bytes(response.read())
            with lock:
                state["captures"] += 1
                state["last_capture_at"] = time.time()
                state["last_error"] = ""
        except Exception as exc:
            with lock:
                state["failures"] += 1
                state["last_error"] = str(exc)
        if stop_event.wait(max(0.0, interval - (time.perf_counter() - started))):
            break
    with lock:
        state["state"] = "stopped"


class ImageCaptureWorker:
    def __init__(self, cfg: dict, url: str):
        self.url = url
        self.polling_rate_ms = cfg["snapshot_polling_rate"]
        self.destination = Path.home() / "Pictures" / "snapshot.jpeg"
        self.manager = None
        self.lock = None
        self.state = None
        self.stop_event = None
        self.process = None

    def start(self, manager) -> None:
        self.manager = manager
        self.lock = manager.Lock()
        self.stop_event = multiprocessing.Event()
        self.state = manager.dict({
            "state": "starting", "captures": 0, "failures": 0,
            "last_capture_at": None, "last_error": "",
        })
        self.process = multiprocessing.Process(
            target=_capture_process,
            args=(self.url, self.polling_rate_ms, str(self.destination), self.state, self.lock, self.stop_event),
            name="image-capture",
            daemon=True,
        )
        self.process.start()

    def stop(self) -> None:
        if self.stop_event:
            self.stop_event.set()

    def join(self, timeout: float) -> None:
        if self.process:
            self.process.join(timeout)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "state": self.state["state"],
                "url": self.url,
                "destination": str(self.destination),
                "snapshot_polling_rate": self.polling_rate_ms,
                "captures": self.state["captures"],
                "failures": self.state["failures"],
                "last_capture_at": self.state["last_capture_at"],
                "last_error": self.state["last_error"],
            }


class ImageCaptureWorkload:
    def __init__(self, cfg: dict, url: str | None):
        capture_cfg = cfg or {}
        self.worker = (
            ImageCaptureWorker(capture_cfg, url)
            if capture_cfg.get("enabled", False) and url
            else None
        )
        self.manager = multiprocessing.Manager() if self.worker else None

    def start(self) -> None:
        if self.worker:
            self.worker.start(self.manager)

    def stop(self) -> None:
        if self.worker:
            self.worker.stop()
            self.worker.join(10)

    def close(self) -> None:
        if self.manager:
            self.manager.shutdown()

    def snapshot(self) -> dict | None:
        return self.worker.snapshot() if self.worker else None