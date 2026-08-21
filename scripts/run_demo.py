"""
Automated Demo Launcher for Agentic ML Engineering Platform.

Workflow:
1. Launches FastAPI backend microservice on port 8000 via uvicorn.
2. Launches Next.js 16 frontend on port 3000 via `npm run dev`.
3. Performs automated health checks on both services.
4. Opens Google Chrome to http://localhost:3000/pipeline.
5. Keeps processes alive and streams real-time status to console.

Usage:
    python scripts/run_demo.py
Or use run_demo.bat / run_demo.ps1 from repo root.
"""

import os
import sys
import time
import subprocess
import webbrowser
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

# Resolve Python interpreter: prefer venv, fall back to system python
VENV_PYTHON = ROOT_DIR / "venv" / "Scripts" / "python.exe"
if not VENV_PYTHON.exists():
    VENV_PYTHON = Path(sys.executable)

BACKEND_URL  = "http://localhost:8000/health"
FRONTEND_URL = "http://localhost:3000"
TARGET_URL   = "http://localhost:3000/pipeline"


def banner(msg: str, char: str = "=", width: int = 70):
    print(char * width)
    print(f"  {msg}")
    print(char * width)


def health_check(url: str, timeout: int = 45, label: str = "Service") -> bool:
    """Poll a URL until HTTP 200 or timeout."""
    deadline = time.time() + timeout
    print(f"  [?] Awaiting {label} at {url} ...", end="", flush=True)
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AgenticML-HealthCheck"})
            with urllib.request.urlopen(req, timeout=3) as r:
                if r.status in (200, 204, 304):
                    elapsed = round(time.time() - (deadline - timeout), 1)
                    print(f"  [ONLINE in {elapsed}s]")
                    return True
        except Exception:
            pass
        time.sleep(1.5)
        print(".", end="", flush=True)
    print(f"  [TIMEOUT after {timeout}s]")
    return False


def open_chrome(url: str):
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        "/usr/bin/google-chrome",
        "/usr/bin/chromium-browser",
    ]
    for path in chrome_paths:
        if os.path.exists(path):
            try:
                subprocess.Popen([path, url])
                print(f"  [+] Google Chrome opened -> {url}")
                return
            except Exception:
                pass
    print("  [*] Chrome not found – falling back to system default browser")
    webbrowser.open(url)


def kill_proc(name: str, proc: subprocess.Popen):
    print(f"  [x] Stopping {name} (PID {proc.pid}) ...", end="")
    try:
        if os.name == "nt":
            subprocess.call(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            proc.terminate()
            proc.wait(timeout=5)
    except Exception:
        pass
    print(" done")


def main():
    banner("AGENTIC-ML AUTONOMOUS DEMO LAUNCHER")
    print(f"\n  Python  : {VENV_PYTHON}")
    print(f"  Root    : {ROOT_DIR}")
    print(f"  Frontend: {FRONTEND_DIR}\n")

    processes: list[tuple[str, subprocess.Popen]] = []

    # ---------------------------------------------------------------
    # 1. Start FastAPI Backend on port 8000
    # ---------------------------------------------------------------
    print("[1/4] Launching FastAPI backend on http://localhost:8000 ...")
    backend_cmd = [
        str(VENV_PYTHON), "-m", "uvicorn",
        "src.agentic_ml.api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--log-level", "warning",
    ]
    try:
        backend_proc = subprocess.Popen(
            backend_cmd,
            cwd=str(ROOT_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        processes.append(("FastAPI Backend", backend_proc))
        print(f"  [+] FastAPI process started (PID {backend_proc.pid})")
    except FileNotFoundError as e:
        print(f"\n  [!] FATAL: Could not start backend: {e}")
        print("       Make sure uvicorn is installed: pip install uvicorn")
        sys.exit(1)

    # ---------------------------------------------------------------
    # 2. Start Next.js frontend on port 3000
    # ---------------------------------------------------------------
    print("\n[2/4] Launching Next.js frontend on http://localhost:3000 ...")
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    try:
        frontend_proc = subprocess.Popen(
            [npm_cmd, "run", "dev"],
            cwd=str(FRONTEND_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        processes.append(("Next.js Frontend", frontend_proc))
        print(f"  [+] Next.js process started (PID {frontend_proc.pid})")
    except FileNotFoundError as e:
        print(f"\n  [!] FATAL: Could not start frontend: {e}")
        print("       Make sure Node.js is installed and `npm install` ran inside /frontend/")
        sys.exit(1)

    # ---------------------------------------------------------------
    # 3. Automated Health Checks
    # ---------------------------------------------------------------
    print("\n[3/4] Running automated health checks ...\n")

    backend_ok = health_check(BACKEND_URL, timeout=40, label="FastAPI Backend")
    if not backend_ok:
        print("\n  [!] FATAL: FastAPI backend did not respond in time.")
        print("       Check if port 8000 is already occupied or a dependency is missing.")
        for name, proc in processes:
            kill_proc(name, proc)
        sys.exit(1)

    frontend_ok = health_check(FRONTEND_URL, timeout=60, label="Next.js Frontend")
    if not frontend_ok:
        print("\n  [!] FATAL: Next.js frontend did not respond in time.")
        print("       Ensure `npm install` was executed inside the /frontend/ directory.")
        for name, proc in processes:
            kill_proc(name, proc)
        sys.exit(1)

    # ---------------------------------------------------------------
    # 4. Open Chrome
    # ---------------------------------------------------------------
    print(f"\n[4/4] Opening live application in browser ...")
    open_chrome(TARGET_URL)

    # ---------------------------------------------------------------
    # Done
    # ---------------------------------------------------------------
    banner("AGENTIC-ML DEMO IS LIVE!", char="*")
    print(f"""
  Pipeline Studio   ->  {TARGET_URL}
  Landing Page      ->  http://localhost:3000
  FastAPI Swagger   ->  http://localhost:8000/docs
  SSE Stream API    ->  http://localhost:8000/api/pipeline/stream
  Download Artifact ->  http://localhost:8000/api/artifacts/download/model.pkl

  Press Ctrl+C to stop all services.
""")

    try:
        while True:
            # Simple watchdog: exit if backend process dies unexpectedly
            if backend_proc.poll() is not None:
                print("\n  [!] FastAPI backend exited unexpectedly. Shutting down.")
                break
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n\n  [*] Ctrl+C detected – shutting down services ...")

    for name, proc in processes:
        kill_proc(name, proc)
    print("\n  [+] All services stopped cleanly. Goodbye!")


if __name__ == "__main__":
    main()
