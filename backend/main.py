"""
backend/main.py  —  FastAPI + python-socketio  |  Edge-AI LPR System  v2.3
===========================================================================

FACTORY PATTERN — WHY THIS MATTERS
────────────────────────────────────
`import socketio` triggers python-engineio which triggers aiohttp.
On Python 3.13 + aiohttp 3.13.x on Windows this can hang for 30-120 s
or deadlock permanently due to a threading-model change in CPython 3.13.

The factory pattern solves this by keeping ALL third-party imports out of
the module's global scope.  Uvicorn steps:
  1. `import backend.main`        → instant (only stdlib here)
  2. `create_app()` is called     → socketio imported here (can take time)
  3. Port is bound & requests accepted

To run:
  python dev.py                   ← preferred
  python -m uvicorn backend.main:create_app --factory --port 8000
"""

# ── Stdlib only at module scope — these are guaranteed instant ───────────────
import asyncio
import os
import sys
import threading
import time

# Force UTF-8 in this subprocess (uvicorn inherits the parent's CP1252 codepage)
import sys as _sys
if _sys.stdout and hasattr(_sys.stdout, 'reconfigure'):
    _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if _sys.stderr and hasattr(_sys.stderr, 'reconfigure'):
    _sys.stderr.reconfigure(encoding='utf-8', errors='replace')
del _sys

print("[BOOT] 1/5: Module imported (stdlib only - no socketio yet).")

# ── Path bootstrap ────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Module-level nulls — populated inside create_app() ───────────────────────
# Using module-level references lets safe_emit() and _ai_worker() close over
# them without needing to be redefined each time create_app() runs.
sio    = None   # socketio.AsyncServer instance
_server = None  # imported server module
_loop  = None   # asyncio event-loop reference for cross-thread emit
_placeholder_jpg: bytes = b""


# ═══════════════════════════════════════════════════════════════════════════════
# Thread-safe emit helper  (safe to call from AI background thread)
# ═══════════════════════════════════════════════════════════════════════════════
def safe_emit(event: str, data: dict):
    """Schedule sio.emit() from any OS thread onto the asyncio event loop."""
    if sio and _loop and not _loop.is_closed():
        asyncio.run_coroutine_threadsafe(sio.emit(event, data), _loop)


# ═══════════════════════════════════════════════════════════════════════════════
# MJPEG video feed
# ═══════════════════════════════════════════════════════════════════════════════
def _build_placeholder() -> bytes:
    import cv2, numpy as np
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Webcam initialising...", (110, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 245, 255), 2)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


async def _frame_generator():
    global _placeholder_jpg
    loop = asyncio.get_event_loop()
    while True:
        jpg = b""
        if _server is not None:
            with _server._frame_lock:
                jpg = _server.shared_frame_jpg
        if not jpg:
            if not _placeholder_jpg:
                _placeholder_jpg = await loop.run_in_executor(None, _build_placeholder)
            jpg = _placeholder_jpg
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        await asyncio.sleep(0.04)


# ═══════════════════════════════════════════════════════════════════════════════
# REST endpoint handlers  (plain async functions — registered in create_app)
# ═══════════════════════════════════════════════════════════════════════════════
async def video_feed():
    from starlette.responses import StreamingResponse
    return StreamingResponse(
        _frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


async def force_open():
    from fastapi.responses import JSONResponse
    if _server:
        _server.dashboard_state["plate_text"] = "FORCE OVERRIDE"
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: _server.open_gate(sio=None)
        )
    ts = time.strftime("%H:%M:%S")
    await sio.emit("gate_event", {"status": "OPEN", "plate": "FORCE OVERRIDE", "timestamp": ts})
    return JSONResponse({"status": "ok"})


async def system_reset():
    from fastapi.responses import JSONResponse
    if _server:
        _server.dashboard_state.update({
            "intensity": 0.0, "lifi_state": False, "pulse_count": 0,
            "gate_status": "IDLE", "plate_text": "", "authorized": False,
        })
    await sio.emit("system_reset", {})
    return JSONResponse({"status": "reset"})


def get_state():
    from fastapi.responses import JSONResponse
    return JSONResponse(_server.dashboard_state if _server else {})


def health():
    from fastapi.responses import JSONResponse
    return JSONResponse({"status": "ok", "server_ready": _server is not None})


# ═══════════════════════════════════════════════════════════════════════════════
# Telemetry bridge  (100 ms asyncio polling loop)
# ═══════════════════════════════════════════════════════════════════════════════
_last_telemetry: dict = {}


async def _telemetry_loop():
    global _last_telemetry
    while True:
        if _server is not None:
            state = _server.dashboard_state.copy()
            if state != _last_telemetry:
                await sio.emit("telemetry_update", {
                    "intensity":   state.get("intensity",   0.0),
                    "pulse_count": state.get("pulse_count", 0),
                    "lifi_state":  state.get("lifi_state",  False),
                    "plate_text":  state.get("plate_text",  ""),
                    "gate_status": state.get("gate_status", "IDLE"),
                    "authorized":  state.get("authorized",  False),
                })
                _last_telemetry = state.copy()
        await asyncio.sleep(0.1)


# ═══════════════════════════════════════════════════════════════════════════════
# AI background worker  (OS thread — never touches the event loop directly)
# ═══════════════════════════════════════════════════════════════════════════════
def _ai_worker():
    global _server
    print("[BOOT] 5/5: AI thread — importing server module...")
    try:
        import server as srv
    except Exception as exc:
        print(f"[BOOT] ✗ FATAL: Could not import server.py — {exc}")
        import traceback; traceback.print_exc()
        return

    _server = srv
    print("[BOOT] 5/5: server module imported OK.")

    # Patch server's emit so all events reach the browser
    def _bridged_emit(sio_ignored, event, data):
        safe_emit(event, data)
    srv._emit = _bridged_emit

    print("[BOOT] 5/5: Launching AI pipeline (camera + YOLO init inside)...")
    try:
        srv.run_ai_pipeline(sio=None)
    except Exception as exc:
        print(f"[AI-WORKER] Pipeline crashed: {exc}")
        import traceback; traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════════
# ASGI FACTORY  ← uvicorn calls this with --factory
# ═══════════════════════════════════════════════════════════════════════════════
def create_app():
    """
    ASGI factory.  All heavy imports (socketio / engineio / aiohttp) happen
    here so the module-level import phase stays instant.

    Uvicorn invocation:
        uvicorn backend.main:create_app --factory --host 0.0.0.0 --port 8000
    """
    global sio

    # ── Step 2: import python-socketio (may be slow on first run) ────────────
    print("[BOOT] 2/5: Importing python-socketio (engineio + aiohttp)...")
    try:
        import socketio as _socketio
        print("[BOOT] 2/5: python-socketio imported OK.")
    except Exception as exc:
        print(f"[BOOT] ✗ FATAL: import socketio failed — {exc}")
        raise

    # ── Step 3: build FastAPI + Socket.IO ────────────────────────────────────
    print("[BOOT] 3/5: Building FastAPI app...")
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="Edge-AI LPR API", version="2.3")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:5000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    sio = _socketio.AsyncServer(
        async_mode="asgi",
        cors_allowed_origins="*",
        logger=False,
        engineio_logger=False,
    )
    print("[BOOT] 3/5: FastAPI + Socket.IO created.")

    # ── Register REST routes ──────────────────────────────────────────────────
    app.add_api_route("/video_feed",    video_feed,    methods=["GET"])
    app.add_api_route("/force-open",    force_open,    methods=["POST"])
    app.add_api_route("/system-reset",  system_reset,  methods=["POST"])
    app.add_api_route("/state",         get_state,     methods=["GET"])
    app.add_api_route("/health",        health,        methods=["GET"])

    # ── Register Socket.IO events ─────────────────────────────────────────────
    @sio.event
    async def connect(sid, environ):
        print(f"[WS] Client connected: {sid}")
        if _server:
            await sio.emit("state_snapshot", _server.dashboard_state, to=sid)

    @sio.event
    async def disconnect(sid):
        print(f"[WS] Client disconnected: {sid}")

    # ── Startup hook ──────────────────────────────────────────────────────────
    @app.on_event("startup")
    async def _startup():
        global _loop
        print("[BOOT] 4/5: FastAPI startup - Uvicorn is live on 0.0.0.0:8000")
        _loop = asyncio.get_event_loop()
        asyncio.create_task(_telemetry_loop())
        ai = threading.Thread(target=_ai_worker, daemon=True, name="AI-Pipeline")
        ai.start()
        print(f"[BOOT] 4/5: AI-Pipeline thread started (TID={ai.ident}).")
        print("[BOOT] [OK] Backend READY - Dashboard -> http://localhost:5173")
        print("[BOOT]      API docs   -> http://localhost:8080/docs")

    # ── Assemble final ASGI app ───────────────────────────────────────────────
    print("[BOOT] 3/5: Assembling socket_app ASGI wrapper...")
    socket_app = _socketio.ASGIApp(sio, other_asgi_app=app)
    print("[BOOT] 3/5: socket_app ready.")
    return socket_app


# ═══════════════════════════════════════════════════════════════════════════════
# Direct entry-point  —  python backend/main.py
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    print("[BOOT] Direct launch (python backend/main.py)...")
    uvicorn.run(
        "backend.main:create_app",
        factory=True,
        host="0.0.0.0",
        port=8080,
        reload=False,
        workers=1,
    )
