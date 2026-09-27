from __future__ import annotations

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent


def command(name: str) -> str:
    if os.name == "nt" and name == "npm":
        return "npm.cmd"
    return name


backend = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "backend.server:app", "--host", "127.0.0.1", "--port", "8765"],
    cwd=ROOT,
)

try:
    time.sleep(1.5)
    frontend = subprocess.Popen(
        [command("npm"), "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"],
        cwd=ROOT / "frontend",
    )
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:5173")
    print("PersonaCluster Studio: http://127.0.0.1:5173")
    print(f"PersonaCluster root: {PROJECT_ROOT}")
    print("Press Ctrl+C to stop Studio.")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    for proc in (locals().get("frontend"), backend):
        if proc and proc.poll() is None:
            proc.terminate()
