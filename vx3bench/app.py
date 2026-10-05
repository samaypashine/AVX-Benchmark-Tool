import argparse,threading,webbrowser
from pathlib import Path
from .webserver import serve
def main():
    p = argparse.ArgumentParser();
    p.add_argument("--host",default="127.0.0.1")
    p.add_argument("--port",type=int,default=8420)
    p.add_argument("--config-dir",type=Path,default=Path("configs"))
    p.add_argument("--reports-dir",type=Path,default=Path("reports"))
    p.add_argument("--no-browser",action="store_true")
    args = p.parse_args()

    if not args.no_browser:threading.Timer(1, lambda:webbrowser.open(f"http://localhost:{args.port}")).start()
    serve(args.host, args.port, args.config_dir, args.reports_dir)
