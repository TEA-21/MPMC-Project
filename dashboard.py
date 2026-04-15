"""
dashboard.py — Cyberpunk Security Dashboard
Flask + Flask-SocketIO entry point.

Run with:  python dashboard.py
Then open:  http://localhost:5000
"""

import threading
import time
import server  # Import our existing AI pipeline

from flask import Flask, Response, jsonify, render_template
from flask_socketio import SocketIO
from flask_cors import CORS

# ── App setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "edge-ai-lpr-secret"
CORS(app)

# Use eventlet/gevent for async SocketIO. Falls back to threading.
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


# ── MJPEG video feed ────────────────────────────────────────────────────────
def _gen_frames():
    """Generator that yields the latest JPEG frame as a multipart stream."""
    PLACEHOLDER = _build_placeholder()
    while True:
        with server._frame_lock:
            jpg = server.shared_frame_jpg
        if jpg:
            frame = jpg
        else:
            frame = PLACEHOLDER
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(0.04)  # ~25 fps cap


def _build_placeholder():
    """Returns a tiny solid-black JPEG for when the webcam hasn't started."""
    import cv2, numpy as np
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Awaiting webcam...", (180, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 245, 255), 2)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        _gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/force-open", methods=["POST"])
def force_open():
    server.dashboard_state["plate_text"] = "FORCE OVERRIDE"
    server.open_gate(sio=socketio)
    return jsonify({"status": "ok"})


@app.route("/system-reset", methods=["POST"])
def system_reset():
    server.dashboard_state.update({
        "intensity":   0.0,
        "lifi_state":  False,
        "pulse_count": 0,
        "gate_status": "IDLE",
        "plate_text":  "",
        "authorized":  False,
    })
    socketio.emit("system_reset", {})
    return jsonify({"status": "reset"})


@app.route("/state")
def state():
    """Snapshot of current dashboard state (for page refresh / initial load)."""
    return jsonify(server.dashboard_state)


# ── Socket.IO events ────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    # Push current state immediately on connect
    socketio.emit("state_snapshot", server.dashboard_state)


# ── Startup ─────────────────────────────────────────────────────────────────
def start_dashboard():
    # Start the AI pipeline in a daemon background thread
    ai_thread = threading.Thread(
        target=server.run_ai_pipeline,
        args=(socketio,),
        daemon=True,
        name="AI-Pipeline",
    )
    ai_thread.start()
    print("[DASHBOARD] AI pipeline thread started.")
    print("[DASHBOARD] Open http://localhost:5000 in your browser.")

    # Run Flask-SocketIO (blocking)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    start_dashboard()
