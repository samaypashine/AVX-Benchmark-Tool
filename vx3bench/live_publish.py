import json, os, random, threading, time
from pathlib import Path
_LOCK = threading.Lock()

def publish_json(path, payload, retries=8):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{random.randrange(1000000)}.tmp")
    last = None
    try:
        temp.write_text(text, encoding="utf-8")
        for attempt in range(retries):
            try:
                os.replace(temp, path)
                return None
            except (PermissionError, OSError) as exc:
                last = exc
                time.sleep(min(0.025 * (attempt + 1), 0.2))
        with _LOCK:
            try:
                path.write_text(text, encoding="utf-8")
                return f"atomic replace unavailable; direct write used: {last}"
            except (PermissionError, OSError) as exc:
                return f"live publication skipped: {exc}"
    finally:
        try: temp.unlink(missing_ok=True)
        except OSError: pass
