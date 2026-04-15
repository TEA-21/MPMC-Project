"""
dev.py — Unified launcher for Edge-AI LPR system.

Starts:
  1. FastAPI backend  (uvicorn)  → http://127.0.0.1:8000
  2. Vite dev server (npm)      → http://localhost:5173  ← open this

All stdout/stderr from both processes is forwarded to this terminal so
crashes (missing libraries, port conflicts, etc.) are immediately visible.

Press Ctrl+C once to shut everything down cleanly.
"""
import subprocess
import sys
import os
import time
import socket
import urllib.request

# Force UTF-8 on Windows (default console is CP1252 which can't render ━ etc.)
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


ROOT     = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.join(ROOT, "frontend")

procs = []


# ── Health-check helpers ────────────────────────────────────────────────────
def _port_open(host, port, timeout=0.5):
    """Return True if something is listening on host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_backend(host="127.0.0.1", port=8080, retries=40, delay=0.5):
    """Block until the backend is accepting TCP connections or retries run out."""
    print(f"[DEV] Waiting for backend on {host}:{port} ", end="", flush=True)
    for _ in range(retries):
        if _port_open(host, port):
            print(" ✓ READY", flush=True)
            return True
        print(".", end="", flush=True)
        time.sleep(delay)
    print(" ✗ TIMEOUT — starting Vite anyway", flush=True)
    return False


# ── Main ────────────────────────────────────────────────────────────────────
def run():
    global procs

    print("\n\033[96m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")
    print("\033[96m  ⚡ EDGE-AI LPR — CYBERPUNK COMMAND CENTER\033[0m")
    print("\033[96m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")
    print("\033[93m  Backend  → http://127.0.0.1:8080\033[0m")
    print("\033[92m  Frontend → http://localhost:5173  (open this)\033[0m")
    print("\033[96m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m\n")

    # ── 1. FastAPI backend via uvicorn ──────────────────────────────────────
    # stdout=None / stderr=None means "inherit from this process" → terminal
    backend_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "backend.main:create_app",
            "--factory",            # use factory pattern: create_app() called after import
            "--host", "0.0.0.0",
            "--port", "8080",
            # --reload is intentionally omitted: uvicorn's file-watcher forks
            # a child process which briefly closes the port, confusing the
            # health check and making the Vite proxy fail.
        ],
        cwd=ROOT,
        stdout=None,   # inherit terminal → tracebacks are visible
        stderr=None,
    )
    procs.append(backend_proc)
    print(f"[DEV] Backend PID {backend_proc.pid} started (uvicorn).")

    # Wait until the backend is actually listening before starting Vite
    _wait_for_backend()

    # Check if uvicorn already crashed (missing lib, port in use, etc.)
    if backend_proc.poll() is not None:
        print(f"\033[91m[DEV] ✗ Backend exited with code {backend_proc.returncode}.\033[0m")
        print("\033[91m[DEV]   Check the output above for the error (missing pip package, etc.)\033[0m")
        shutdown()
        return

    # ── 2. Vite dev server ──────────────────────────────────────────────────
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=FRONTEND,
        stdout=None,   # ← inherit terminal
        stderr=None,   # ← inherit terminal
    )
    procs.append(frontend_proc)
    print(f"[DEV] Frontend PID {frontend_proc.pid} started (Vite).\n")

    try:
        # Loop until any process exits unexpectedly
        while True:
            for p in procs:
                rc = p.poll()
                if rc is not None:
                    name = "Backend" if p is procs[0] else "Frontend"
                    print(f"\033[91m[DEV] {name} exited with code {rc}.\033[0m")
                    raise SystemExit(rc)
            time.sleep(0.5)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        shutdown()


def shutdown():
    print("\n[DEV] Shutting down all services...")
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    print("[DEV] All services stopped. Goodbye!")


if __name__ == "__main__":
    run()
