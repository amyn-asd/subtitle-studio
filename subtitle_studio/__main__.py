from __future__ import annotations

import argparse
import json
import os
import sys
import socket
import subprocess
import threading
import time
import webbrowser

import uvicorn

from .app import create_app
from .config import DATA, ROOT, initialize
from .media import NO_WINDOW
from .subtitles import atomic_text


def main():
    parser = argparse.ArgumentParser(description="Subtitle Studio local server")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    initialize()
    app = create_app()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", args.port))
    sock.listen(128)
    port = sock.getsockname()[1]
    atomic_text(DATA / "server.json", json.dumps({"port": port, "token": app.state.token, "pid": os.getpid()}))
    if not args.no_browser:
        def open_browser():
            time.sleep(1.5)
            webbrowser.open(f"http://127.0.0.1:{port}")
        threading.Thread(target=open_browser, daemon=True).start()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    restarting = threading.Event()
    def shutdown(restart=False):
        if restart:
            restarting.set()
        def later():
            time.sleep(.5)
            server.should_exit = True
        threading.Thread(target=later, daemon=True).start()
    app.state.shutdown = shutdown
    print(f"Subtitle Studio is running at http://127.0.0.1:{port}", flush=True)
    try:
        server.run(sockets=[sock])
    finally:
        app.state.jobs.close()
        sock.close()
    if restarting.is_set():
        # Windows execv does not safely quote an interpreter path containing spaces.
        # Popen's argument list does, and lets the new server outlive the launcher.
        with (DATA / "logs" / "server-output.log").open("a", encoding="utf-8") as out, \
             (DATA / "logs" / "server-error.log").open("a", encoding="utf-8") as err:
            subprocess.Popen([sys.executable, "-m", "subtitle_studio", "--port", str(port), "--no-browser"],
                             cwd=ROOT, stdout=out, stderr=err, creationflags=NO_WINDOW)


if __name__ == "__main__":
    main()
