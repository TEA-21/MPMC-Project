"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   demo_mode.py  —  Edge-AI LPR System  |  Presentation HUD                 ║
║   Author : Senior CV Engineer                                                ║
║   Purpose: Live webcam demonstration of all ML features with on-screen HUD  ║
║                                                                              ║
║   CONTROLS:                                                                  ║
║     [SPACE]  — Simulate ESP32-CAM hardware trigger (runs heavy ML)           ║
║     [Q]      — Quit                                                          ║
║     [C]      — Clear last result / reset state to WAITING                   ║
║     [L]      — Manually inject a Li-Fi test signal (for demo without torch)  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Standard Library ────────────────────────────────────────────────────────
import cv2
import numpy as np
import time
import datetime
import csv
import threading
import collections
import warnings
import os

# ── Optional: SciPy for FFT-based Li-Fi detection ───────────────────────────
try:
    import scipy.signal
    from scipy.fft import rfft, rfftfreq
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    warnings.warn(
        "scipy not installed — Li-Fi FFT disabled. Run: pip install scipy",
        RuntimeWarning, stacklevel=1,
    )

# ── Optional: Pytesseract for OCR ───────────────────────────────────────────
try:
    import pytesseract
    _TESS_AVAILABLE = True
    # On Windows, set path if not in PATH. Adjust if installed elsewhere:
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except ImportError:
    _TESS_AVAILABLE = False
    warnings.warn(
        "pytesseract not installed — OCR will be simulated. Run: pip install pytesseract",
        RuntimeWarning, stacklevel=1,
    )

# ── Optional: YOLOv8 for vehicle detection & face detection ─────────────────
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    warnings.warn(
        "ultralytics not installed — YOLO will be simulated. Run: pip install ultralytics",
        RuntimeWarning, stacklevel=1,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Authorised plates → (expected_color, expected_type)
AUTHORIZED_DB = {
    "TN01AB1234": ("White", "SUV"),
    "KA05XY9876": ("Red",   "Sedan"),
}

# Li-Fi target flash frequency (Hz) — "3 flashes per second"
LIFI_TARGET_FREQ   = 3.0
LIFI_FREQ_TOLERANCE = 0.8   # ±0.8 Hz
LIFI_ROLLING_WINDOW = 60    # frames

# CSV log file
CSV_LOG = "parking_log.csv"

# Dashboard dimensions (left side overlay)
DASH_WIDTH = 310
DASH_ALPHA = 0.75   # transparency of semi-transparent black panel

# Colours (BGR)
C_GREEN   = (0,   230,  80)
C_BLUE    = (60,  140, 255)
C_RED     = (50,   50, 255)
C_YELLOW  = (0,   210, 255)
C_CYAN    = (255, 210,   0)
C_WHITE   = (240, 240, 240)
C_GRAY    = (120, 120, 120)
C_ORANGE  = (0,   160, 255)
C_PURPLE  = (200,  60, 180)
C_BLACK   = (0,     0,   0)

# Fonts
FONT      = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO = cv2.FONT_HERSHEY_PLAIN

# YOLO COCO vehicle class IDs
VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
VEHICLE_TYPE_MAP = {
    "suv": [2], "sedan": [2], "hatchback": [2], "car": [2],
    "truck": [7], "pickup": [7], "motorcycle": [3], "bike": [3],
}


# ═══════════════════════════════════════════════════════════════════════════
#  GLOBAL STATE  (all thread-safe via threading.Lock)
# ═══════════════════════════════════════════════════════════════════════════

class AppState:
    """Central mutable state shared between CV loop and background workers."""
    def __init__(self):
        self._lock = threading.Lock()

        # System state machine
        self.system_state   = "WAITING"   # WAITING | PROCESSING | GRANTED | DENIED | LIFI_UNLOCK
        self.state_changed  = time.time()

        # OCR result
        self.detected_plate = None     # e.g. "TN01AB1234"
        self.plate_boxes    = []       # [(x,y,w,h), …]   from contour stage

        # Vehicle detection
        self.vehicle_boxes  = []       # [(x1,y1,x2,y2,label,conf), …]
        self.vehicle_color  = None     # e.g. "Red"

        # Face detection
        self.face_boxes     = []       # [(x,y,w,h), …]

        # Li-Fi
        self.lifi_status    = "LISTENING"  # LISTENING | PATTERN DETECTED
        self.lifi_last_hit  = 0.0

        # Counters
        self.total_entries  = 0
        self.session_grants = 0
        self.session_denies = 0

        # Last access result message
        self.access_msg     = ""
        self.access_color   = C_WHITE

        # Frame rate
        self.fps            = 0.0

    def set(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)
            if "system_state" in kwargs:
                self.state_changed = time.time()

    def get_state(self):
        with self._lock:
            return self.system_state

STATE = AppState()


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL REGISTRY  — Lazy-loaded singletons
# ═══════════════════════════════════════════════════════════════════════════

_models = {}
_model_lock = threading.Lock()


def _load_models():
    """Bootstrap all heavy models exactly once in a background thread."""
    global _models
    with _model_lock:
        if _models.get("loaded"):
            return

        print("[DEMO] Loading models…")

        # ── Haar face cascade (always available via OpenCV) ──────────────
        _models["haar_face"] = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        _models["haar_plate"] = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
        )
        print("[DEMO]   ✓ Haar cascades loaded (face + plate).")

        # ── YOLOv8 object detection (vehicles) ──────────────────────────
        if _YOLO_AVAILABLE:
            try:
                _models["yolo_obj"] = YOLO("yolov8n.pt")
                print("[DEMO]   ✓ YOLOv8n object detection loaded.")
            except Exception as e:
                print(f"[DEMO]   ✗ YOLOv8n load failed: {e}")
                _models["yolo_obj"] = None

            # ── YOLOv8-face (optional, falls back to Haar) ──────────────
            try:
                _models["yolo_face"] = YOLO("yolov8n-face.pt")
                print("[DEMO]   ✓ YOLOv8-face loaded.")
            except Exception as e:
                print(f"[DEMO]   ✗ YOLOv8-face load failed ({e}). Haar fallback active.")
                _models["yolo_face"] = None
        else:
            _models["yolo_obj"]  = None
            _models["yolo_face"] = None

        _models["loaded"] = True
        print("[DEMO] All models ready.")


# Start model loading immediately in background
threading.Thread(target=_load_models, daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE 1 — OCR PLATE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def _find_plate_candidates(frame):
    """
    Returns a list of (x,y,w,h) bounding boxes for likely license plate regions
    using a two-stage edge + contour filter with aspect-ratio gating.
    """
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes  = []

    # Stage 1 — Haar cascade
    haar = _models.get("haar_plate")
    if haar and not haar.empty():
        dets = haar.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4,
            minSize=(60, 20), flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(dets) > 0:
            boxes.extend([(x, y, w, h) for (x, y, w, h) in dets])

    # Stage 2 — Contour + aspect-ratio gate (2.0 ≤ ratio ≤ 6.5)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 200)
    cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        aspect = w / h
        area   = w * h
        if 2.0 <= aspect <= 6.5 and 1200 <= area <= 60000:
            boxes.append((x, y, w, h))

    return boxes


def run_ocr(frame):
    """
    Runs plate detection + OCR on the frame.
    Returns: (plate_text | None, [(x,y,w,h) candidates])
    """
    candidates = _find_plate_candidates(frame)

    if not _TESS_AVAILABLE:
        # ── SIMULATION: pick the largest candidate and fake a plate string ──
        if not candidates:
            return None, []
        best = max(candidates, key=lambda b: b[2] * b[3])
        simulated = "TN01AB1234"   # demo plate — change to test DENIED path
        print(f"[OCR] ⚠ Tesseract not installed. Simulating plate: {simulated}")
        return simulated, [best]

    # ── Real OCR ─────────────────────────────────────────────────────────
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    best_text = None
    best_conf = 0.0

    for (x, y, w, h) in candidates[:5]:   # cap at 5 candidates for speed
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        # Pre-process ROI for better OCR
        roi_up   = cv2.resize(roi, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
        _, roi_t = cv2.threshold(roi_up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        raw = pytesseract.image_to_string(
            roi_t,
            config="--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        ).strip()
        # Remove spaces / newlines
        cleaned = "".join(raw.split())
        if len(cleaned) >= 5:
            # Prefer longer strings (more plate characters extracted)
            if len(cleaned) > best_conf:
                best_conf = len(cleaned)
                best_text = cleaned

    return best_text, candidates


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE 2 — MULTI-FACTOR AUTH  (YOLO Vehicle + HSV Color)
# ═══════════════════════════════════════════════════════════════════════════

_HSV_COLOR_RANGES = {
    "Red":    [(np.array([0,70,50]),   np.array([10,255,255])),
               (np.array([170,70,50]), np.array([180,255,255]))],
    "Blue":   [(np.array([100,150,0]), np.array([140,255,255]))],
    "Green":  [(np.array([35,50,50]),  np.array([85,255,255]))],
    "Yellow": [(np.array([20,100,100]),np.array([30,255,255]))],
    "White":  [(np.array([0,0,200]),   np.array([180,30,255]))],
    "Black":  [(np.array([0,0,0]),     np.array([180,255,50]))],
    "Silver": [(np.array([0,0,50]),    np.array([180,30,200]))],
}


def _get_dominant_color(roi):
    """Returns the dominant color name for an image ROI using HSV thresholding."""
    if roi.size == 0:
        return "Unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    best_color, max_px = "Unknown", 0
    for name, ranges in _HSV_COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lo, hi) in ranges:
            mask |= cv2.inRange(hsv, lo, hi)
        px = cv2.countNonZero(mask)
        if px > max_px:
            max_px  = px
            best_color = name
    return best_color


def run_yolo_detection(frame):
    """
    Runs YOLOv8 on the frame to detect vehicles.
    Returns: list of (x1,y1,x2,y2,label_str,conf) for every vehicle found.
    Also populates STATE.vehicle_color with the dominant color of the largest vehicle.
    """
    model = _models.get("yolo_obj")

    if model is None:
        # ── SIMULATION ──────────────────────────────────────────────────
        h, w = frame.shape[:2]
        sim_box = (int(w*0.15), int(h*0.2), int(w*0.85), int(h*0.9), "Car [SIM]", 0.91)
        STATE.set(vehicle_color="White")
        print("[YOLO] ⚠ YOLO not loaded. Simulating vehicle detection.")
        return [sim_box]

    results = model.predict(frame, conf=0.35, verbose=False)
    detections = []
    max_area   = 0
    best_box   = None

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id not in VEHICLE_CLASSES:
                continue
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = f"{VEHICLE_CLASSES[cls_id]} ({conf:.0%})"
            detections.append((x1, y1, x2, y2, label, conf))
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_box = (x1, y1, x2, y2)

    # Extract dominant color from the largest vehicle
    if best_box:
        x1, y1, x2, y2 = best_box
        w, h = x2 - x1, y2 - y1
        cx, cy   = x1 + w // 2, y1 + h // 2
        cw, ch   = int(w * 0.4), int(h * 0.4)
        rx1 = max(0, cx - cw // 2)
        ry1 = max(0, cy - ch // 2 + int(h * 0.1))
        rx2 = min(frame.shape[1], rx1 + cw)
        ry2 = min(frame.shape[0], ry1 + ch)
        roi = frame[ry1:ry2, rx1:rx2]
        color = _get_dominant_color(roi)
        STATE.set(vehicle_color=color)

    return detections


def verify_multi_factor(frame, plate_text):
    """
    Validates YOLO-detected vehicle type + HSV color against the AUTHORIZED_DB.
    Returns True if both factors match, False otherwise.
    """
    expected_color, expected_type = AUTHORIZED_DB.get(plate_text, (None, None))
    if not expected_color:
        return False   # Unknown plate

    detections = run_yolo_detection(frame)
    STATE.set(vehicle_boxes=detections)

    if not detections:
        print("[AUTH] ❌ No vehicle detected by YOLO.")
        return False

    detected_color = STATE.vehicle_color or "Unknown"

    # Map expected type → YOLO class IDs
    expected_ids = VEHICLE_TYPE_MAP.get(expected_type.lower(), [2])
    # Check if any detected vehicle matches expected type
    type_ok = any(
        # label starts with VEHICLE_CLASSES value
        any(VEHICLE_CLASSES.get(cid, "") in det[4] for cid in expected_ids)
        for det in detections
    ) if _YOLO_AVAILABLE else True   # skip type check in simulation mode

    color_ok = (detected_color.lower() == expected_color.lower())

    if not color_ok:
        print(f"[AUTH] ❌ Color Mismatch: Expected {expected_color}, got {detected_color}")
        return False
    if not type_ok:
        print(f"[AUTH] ❌ Type Mismatch: Expected {expected_type}")
        return False

    print(f"[AUTH] ✅ Visual Auth Passed: {detected_color} {expected_type}")
    return True


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE 3 — PRIVACY ANONYMIZATION  (Haar / YOLOv8-face + blur)
# ═══════════════════════════════════════════════════════════════════════════

def detect_faces(frame):
    """
    Detects faces using YOLOv8-face (preferred) or Haar cascade (fallback).
    Returns list of (x,y,w,h) bounding boxes.
    """
    h0, w0  = frame.shape[:2]
    scale   = min(1.0, 640 / w0)
    small   = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)))
    boxes   = []

    yolo_face = _models.get("yolo_face")
    haar_face = _models.get("haar_face")

    if yolo_face is not None:
        results = yolo_face.predict(small, imgsz=640, conf=0.45, iou=0.45, verbose=False)
        for r in results:
            for box in r.boxes.xyxy.tolist():
                x1, y1, x2, y2 = (int(v / scale) for v in box[:4])
                boxes.append((x1, y1, x2 - x1, y2 - y1))
    elif haar_face and not haar_face.empty():
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        dets = haar_face.detectMultiScale(gray, 1.1, 5, minSize=(24, 24))
        for (fx, fy, fw, fh) in (dets if len(dets) > 0 else []):
            boxes.append((
                int(fx / scale), int(fy / scale),
                int(fw / scale), int(fh / scale),
            ))

    return boxes


def apply_face_blur(frame, face_boxes):
    """
    Applies heavy pixelation (mosaic) over each detected face.
    Pixelation is harder to reverse than Gaussian blur.
    """
    out = frame   # mutate in-place for speed in demo
    for (x, y, w, h) in face_boxes:
        # Expand box slightly
        pad = 20
        H, W = out.shape[:2]
        x  = max(0, x - pad)
        y  = max(0, y - pad)
        w  = min(W - x, w + pad * 2)
        h  = min(H - y, h + pad * 2)
        if w < 4 or h < 4:
            continue
        roi  = out[y:y+h, x:x+w]
        tiny = cv2.resize(roi, (16, 16), interpolation=cv2.INTER_LINEAR)
        out[y:y+h, x:x+w] = cv2.resize(tiny, (w, h), interpolation=cv2.INTER_NEAREST)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE 4 — LI-FI AUTHENTICATION  (FFT flashlight detection)
# ═══════════════════════════════════════════════════════════════════════════

class LiFiDetector:
    """Rolling-window FFT flashlight pattern detector (3 Hz target)."""

    def __init__(self):
        self.history = collections.deque(maxlen=LIFI_ROLLING_WINDOW)

    def feed(self, frame):
        """Call every frame. Returns True when the target flash-pattern is confirmed."""
        if not _SCIPY_AVAILABLE:
            return False

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        _, mv, _, _ = cv2.minMaxLoc(blurred)

        self.history.append((time.time(), mv))

        if len(self.history) < 20:
            return False

        times = np.array([t for t, _ in self.history])
        vals  = np.array([v for _, v in self.history])
        duration = times[-1] - times[0]

        if duration < 2.0:
            return False

        fps_est = len(self.history) / duration
        vals_d  = scipy.signal.detrend(vals)

        if np.std(vals_d) < 15:   # scene is static — no flashing
            return False

        N  = len(vals_d)
        T  = 1.0 / fps_est
        yf = rfft(vals_d)
        xf = rfftfreq(N, T)

        mags     = np.abs(yf)
        peak_idx = np.argmax(mags[1:]) + 1
        peak_f   = xf[peak_idx]
        peak_m   = mags[peak_idx]

        if peak_m > 100 and abs(peak_f - LIFI_TARGET_FREQ) <= LIFI_FREQ_TOLERANCE:
            print(f"[LIFI] 🔦 Pattern detected! Freq={peak_f:.2f} Hz  Mag={peak_m:.1f}")
            self.history.clear()
            return True

        return False


_lifi = LiFiDetector()


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE 5 — PREDICTIVE ANALYTICS LOGGING  (CSV)
# ═══════════════════════════════════════════════════════════════════════════

def log_entry_event(plate_text, access_result):
    """Appends one entry event row to parking_log.csv."""
    first_write = not os.path.exists(CSV_LOG) or os.path.getsize(CSV_LOG) == 0
    with open(CSV_LOG, mode="a", newline="") as f:
        writer = csv.writer(f)
        if first_write:
            writer.writerow(["timestamp", "plate", "result"])
        writer.writerow([
            datetime.datetime.now().isoformat(),
            plate_text,
            access_result,
        ])


# ═══════════════════════════════════════════════════════════════════════════
#  HUD DRAWING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def _draw_filled_rect_alpha(frame, x1, y1, x2, y2, color, alpha):
    """Draws a semi-transparent filled rectangle over frame in-place."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _label_box(frame, x1, y1, x2, y2, text, box_color, text_color=C_BLACK, thickness=2):
    """Draws a bounding box with a label above it (pill-style background)."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

    # Label pill
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
    pill_y1 = max(y1 - th - 10, 0)
    pill_y2 = y1
    cv2.rectangle(frame, (x1, pill_y1), (x1 + tw + 8, pill_y2), box_color, -1)
    cv2.putText(frame, text, (x1 + 4, pill_y2 - 4),
                FONT, 0.55, text_color, 1, cv2.LINE_AA)


def _put_dashboard_text(frame, line, text, value, value_color):
    """Places a key=value pair on the dashboard at the given line index."""
    y = 90 + line * 32
    # Key
    cv2.putText(frame, text, (12, y), FONT, 0.48, C_GRAY, 1, cv2.LINE_AA)
    # Value
    cv2.putText(frame, value, (12, y + 18), FONT, 0.55, value_color, 1, cv2.LINE_AA)


def _section_header(frame, y, title):
    """Renders a section divider on the dashboard."""
    cv2.line(frame, (8, y - 5), (DASH_WIDTH - 8, y - 5), (50, 50, 50), 1)
    cv2.putText(frame, title, (12, y + 10), FONT_MONO, 0.9, C_CYAN, 1, cv2.LINE_AA)


def draw_hud(frame, fps):
    """
    Composites the full HUD onto 'frame' in-place:
      - Semi-transparent dashboard panel (left)
      - Live system metrics
      - Bounding boxes for plates, vehicles, faces
      - FPS indicator
    """
    H, W = frame.shape[:2]

    # ── 1. Semi-transparent dashboard background ─────────────────────────
    _draw_filled_rect_alpha(frame, 0, 0, DASH_WIDTH, H, C_BLACK, DASH_ALPHA)

    # ── 2. Header ────────────────────────────────────────────────────────
    cv2.putText(frame, "EDGE-AI  LPR  SYSTEM", (10, 28),
                FONT, 0.60, C_CYAN, 1, cv2.LINE_AA)
    cv2.putText(frame, "Demo Mode  |  v1.0", (10, 50),
                FONT_MONO, 1.0, C_GRAY, 1, cv2.LINE_AA)
    cv2.line(frame, (8, 58), (DASH_WIDTH - 8, 58), C_CYAN, 1)

    # ── 3. System State ──────────────────────────────────────────────────
    state = STATE.system_state
    state_color = {
        "WAITING":     C_GRAY,
        "PROCESSING":  C_YELLOW,
        "GRANTED":     C_GREEN,
        "DENIED":      C_RED,
        "LIFI_UNLOCK": C_PURPLE,
    }.get(state, C_WHITE)

    state_labels = {
        "WAITING":     "Waiting for HW Trigger...",
        "PROCESSING":  "Processing...",
        "GRANTED":     "ACCESS GRANTED",
        "DENIED":      "ACCESS DENIED",
        "LIFI_UNLOCK": "Li-Fi Override Active",
    }

    _section_header(frame, 68, "SYSTEM STATE")
    cv2.putText(frame, state_labels.get(state, state), (12, 104),
                FONT, 0.58, state_color, 1, cv2.LINE_AA)

    # State age indicator (how long in current state)
    elapsed = time.time() - STATE.state_changed
    cv2.putText(frame, f"  {elapsed:.1f}s ago", (12, 120),
                FONT_MONO, 0.85, C_GRAY, 1, cv2.LINE_AA)

    # ── 4. Li-Fi Sensor ──────────────────────────────────────────────────
    _section_header(frame, 135, "LI-FI SENSOR")
    lifi_state = STATE.lifi_status
    lifi_color = C_PURPLE if lifi_state == "PATTERN DETECTED" else C_GRAY
    lifi_prefix = "◉ " if lifi_state == "PATTERN DETECTED" else "◌ "
    cv2.putText(frame, lifi_prefix + lifi_state, (12, 171),
                FONT, 0.55, lifi_color, 1, cv2.LINE_AA)

    # ── 5. Privacy Filter ────────────────────────────────────────────────
    _section_header(frame, 186, "PRIVACY FILTER")
    n_faces = len(STATE.face_boxes)
    cv2.putText(frame, "ACTIVE", (12, 222),
                FONT, 0.55, C_GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"  Faces blurred: {n_faces}", (12, 239),
                FONT_MONO, 0.90, C_GRAY, 1, cv2.LINE_AA)

    # ── 6. OCR Result ────────────────────────────────────────────────────
    _section_header(frame, 254, "OCR / PLATE")
    plate = STATE.detected_plate or "---"
    in_db = plate in AUTHORIZED_DB
    p_color = C_GREEN if in_db else (C_RED if plate != "---" else C_GRAY)
    cv2.putText(frame, plate, (12, 290),
                FONT, 0.68, p_color, 1, cv2.LINE_AA)
    if plate != "---":
        db_label = "IN DATABASE ✓" if in_db else "NOT IN DATABASE ✗"
        cv2.putText(frame, db_label, (12, 308),
                    FONT_MONO, 0.85, p_color, 1, cv2.LINE_AA)

    # ── 7. Vehicle Auth ──────────────────────────────────────────────────
    _section_header(frame, 323, "VEHICLE AUTH")
    v_color = STATE.vehicle_color or "---"
    cv2.putText(frame, f"Color: {v_color}", (12, 359),
                FONT, 0.50, C_BLUE, 1, cv2.LINE_AA)
    n_vehicles = len(STATE.vehicle_boxes)
    cv2.putText(frame, f"Vehicles detected: {n_vehicles}", (12, 376),
                FONT_MONO, 0.85, C_GRAY, 1, cv2.LINE_AA)

    # ── 8. Session Stats ─────────────────────────────────────────────────
    _section_header(frame, 391, "SESSION STATS")
    cv2.putText(frame, f"Granted : {STATE.session_grants}", (12, 427),
                FONT, 0.50, C_GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Denied  : {STATE.session_denies}", (12, 444),
                FONT, 0.50, C_RED,   1, cv2.LINE_AA)
    cv2.putText(frame, f"Total   : {STATE.session_grants + STATE.session_denies}",
                (12, 461), FONT, 0.50, C_WHITE, 1, cv2.LINE_AA)

    # ── 9. Controls hint (bottom of panel) ───────────────────────────────
    hint_y = H - 60
    cv2.line(frame, (8, hint_y - 8), (DASH_WIDTH - 8, hint_y - 8), (50, 50, 50), 1)
    for i, hint in enumerate(["[SPACE] Trigger ML", "[L] Inject Li-Fi", "[C] Clear  [Q] Quit"]):
        cv2.putText(frame, hint, (12, hint_y + i * 18),
                    FONT_MONO, 0.80, C_GRAY, 1, cv2.LINE_AA)

    # ── 10. FPS counter (top-right) ───────────────────────────────────────
    fps_text = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(fps_text, FONT_MONO, 1.0, 1)
    cv2.putText(frame, fps_text, (W - tw - 10, 22),
                FONT_MONO, 1.0, C_YELLOW, 1, cv2.LINE_AA)

    # ── 11. Overlay bounding boxes ────────────────────────────────────────

    # Plate boxes (green)
    for (x, y, w, h) in STATE.plate_boxes:
        label = STATE.detected_plate or "Plate?"
        _label_box(frame, x, y, x + w, y + h, label, C_GREEN)

    # Vehicle boxes (blue)
    for det in STATE.vehicle_boxes:
        x1, y1, x2, y2, label, _ = det
        _label_box(frame, x1, y1, x2, y2,
                   f"{label} | {STATE.vehicle_color or '?'}",
                   C_BLUE, text_color=C_WHITE)

    # ── 12. Access result banner (centre of frame, fades after 3s) ──────
    if STATE.access_msg and (time.time() - STATE.state_changed < 4.0):
        (tw, th), _ = cv2.getTextSize(STATE.access_msg, FONT, 1.3, 2)
        bx = (W - tw) // 2 - 20
        by = H // 2 - th - 20
        _draw_filled_rect_alpha(frame, bx, by, bx + tw + 40, by + th + 40,
                                C_BLACK, 0.70)
        cv2.putText(frame, STATE.access_msg,
                    ((W - tw) // 2, H // 2),
                    FONT, 1.3, STATE.access_color, 2, cv2.LINE_AA)

    return frame


# ═══════════════════════════════════════════════════════════════════════════
#  PROCESSING PIPELINE  (runs on SPACE keypress in a background thread)
# ═══════════════════════════════════════════════════════════════════════════

def run_ml_pipeline(frame_snapshot):
    """
    Full ML pipeline triggered by the simulated hardware event (Spacebar).

    Pipeline:
      1. OCR → extract plate text
      2. Face detection → blur + update state
      3. Database lookup → is plate authorised?
      4. YOLO + Color Auth → multi-factor verification
      5. Log to CSV
      6. Update STATE with result
    """
    STATE.set(system_state="PROCESSING",
              detected_plate=None, plate_boxes=[],
              vehicle_boxes=[], vehicle_color=None, face_boxes=[])

    # ── Step 1: Detect & blur faces (privacy) ─────────────────────────
    print("[PIPELINE] Step 1/4 → Privacy filter (face detection)…")
    face_boxes = detect_faces(frame_snapshot)
    STATE.set(face_boxes=face_boxes)
    if face_boxes:
        print(f"[PRIVACY] Blurring {len(face_boxes)} face(s).")

    # ── Step 2: OCR plate extraction ──────────────────────────────────
    print("[PIPELINE] Step 2/4 → OCR plate extraction…")
    plate_text, candidates = run_ocr(frame_snapshot)
    STATE.set(detected_plate=plate_text, plate_boxes=candidates)

    if not plate_text:
        print("[OCR] ✗ No plate text detected.")
        STATE.set(system_state="DENIED",
                  access_msg="NO PLATE DETECTED",
                  access_color=C_ORANGE,
                  session_denies=STATE.session_denies + 1)
        return

    print(f"[OCR] ✓ Detected plate: {plate_text}")

    # ── Step 3: Database lookup ──────────────────────────────────────
    print("[PIPELINE] Step 3/4 → Database lookup…")
    if plate_text not in AUTHORIZED_DB:
        print(f"[ACCESS DENIED] Plate Mismatch — '{plate_text}' not in DB.")
        STATE.set(system_state="DENIED",
                  access_msg="ACCESS DENIED",
                  access_color=C_RED,
                  session_denies=STATE.session_denies + 1)
        log_entry_event(plate_text, "DENIED-NOT_IN_DB")
        return

    # ── Step 4: Multi-factor visual auth (YOLO + Color) ──────────────
    print("[PIPELINE] Step 4/4 → Multi-factor visual auth (YOLO + Color)…")
    auth_passed = verify_multi_factor(frame_snapshot, plate_text)

    if not auth_passed:
        print(f"[ACCESS DENIED] Visual Auth Failed for plate '{plate_text}'.")
        STATE.set(system_state="DENIED",
                  access_msg="AUTH FAILED",
                  access_color=C_RED,
                  session_denies=STATE.session_denies + 1)
        log_entry_event(plate_text, "DENIED-AUTH_FAIL")
        return

    # ── All checks passed ────────────────────────────────────────────
    print(f"[ACCESS GRANTED] Gate Opening... Plate: {plate_text}")
    STATE.set(system_state="GRANTED",
              access_msg="ACCESS GRANTED",
              access_color=C_GREEN,
              session_grants=STATE.session_grants + 1)
    log_entry_event(plate_text, "GRANTED")

    # Simulate gate open delay then return to WAITING
    time.sleep(5)
    STATE.set(system_state="WAITING",
              detected_plate=None, plate_boxes=[],
              vehicle_boxes=[], vehicle_color=None,
              access_msg="", face_boxes=[])


def lifi_grant():
    """Grants access via Li-Fi override (runs in its own thread)."""
    print("[ACCESS GRANTED] Li-Fi Override — Gate Opening...")
    STATE.set(system_state="LIFI_UNLOCK",
              lifi_status="PATTERN DETECTED",
              access_msg="LI-FI UNLOCK",
              access_color=C_PURPLE,
              session_grants=STATE.session_grants + 1)
    log_entry_event("LIFI_OVERRIDE", "GRANTED-LIFI")
    time.sleep(5)
    STATE.set(system_state="WAITING",
              lifi_status="LISTENING",
              access_msg="", face_boxes=[],
              vehicle_boxes=[], plate_boxes=[])


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN DEMO LOOP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  EDGE-AI LPR DEMO MODE  |  Press [SPACE] to trigger ML")
    print("  [L] inject Li-Fi  |  [C] clear  |  [Q] quit")
    print("=" * 65)

    # ── Camera init ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam (cv2.VideoCapture(0)).")
        return

    # Prefer 720p for a comfortable HUD layout
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_FPS,           30)

    pipeline_thread = None   # handle to the latest ML thread
    fps_timer  = time.time()
    fps_frames = 0
    current_fps = 0.0

    print("[DEMO] Webcam opened. Starting HUD loop…")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame capture failed.")
            break

        # ── FPS calculation ───────────────────────────────────────────
        fps_frames += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 0.5:
            current_fps = fps_frames / elapsed
            fps_frames  = 0
            fps_timer   = time.time()

        # ── Li-Fi: continuous, independent of spacebar ─────────────────
        if STATE.system_state == "WAITING":
            if _lifi.feed(frame):
                STATE.set(lifi_status="PATTERN DETECTED")
                t = threading.Thread(target=lifi_grant, daemon=True)
                t.start()
            else:
                if STATE.lifi_status != "PATTERN DETECTED":
                    STATE.set(lifi_status="LISTENING")

        # ── Face blur: apply in real-time (always on) ─────────────────
        if STATE.face_boxes:
            frame = apply_face_blur(frame, STATE.face_boxes)

        # ── Draw HUD ──────────────────────────────────────────────────
        frame = draw_hud(frame, current_fps)

        cv2.imshow("Edge-AI LPR  |  Demo Mode", frame)

        # ── Key handling ──────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[DEMO] Quit requested.")
            break

        elif key == ord('c'):
            # Clear state
            STATE.set(system_state="WAITING",
                      detected_plate=None, plate_boxes=[],
                      vehicle_boxes=[], vehicle_color=None,
                      face_boxes=[], access_msg="")
            print("[DEMO] State cleared.")

        elif key == ord('l'):
            # Manually inject a Li-Fi event (for presenters without a torch)
            if STATE.system_state == "WAITING":
                print("[DEMO] Manual Li-Fi event injected.")
                t = threading.Thread(target=lifi_grant, daemon=True)
                t.start()

        elif key == ord(' '):
            # ── SPACEBAR — Simulate ESP32 hardware trigger ─────────────
            if STATE.system_state == "WAITING":
                if pipeline_thread is None or not pipeline_thread.is_alive():
                    snap = frame.copy()
                    print("[DEMO] ▶ Hardware trigger simulated — launching ML pipeline…")
                    pipeline_thread = threading.Thread(
                        target=run_ml_pipeline, args=(snap,), daemon=True
                    )
                    pipeline_thread.start()
                else:
                    print("[DEMO] Pipeline already running — wait for result.")
            else:
                print(f"[DEMO] System busy ({STATE.system_state}). Press [C] to reset.")

    cap.release()
    cv2.destroyAllWindows()
    print("[DEMO] Session ended.")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
