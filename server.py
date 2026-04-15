import cv2
import pytesseract
import numpy as np
import urllib.request
import requests
import time
import datetime
import csv
import threading
import threading
import warnings
import collections

# ── Optional SciPy for Li-Fi FFT (install with: pip install scipy) ──
try:
    import scipy.signal
    from scipy.fft import rfft, rfftfreq
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    warnings.warn(
        "scipy not installed – Li-Fi FFT unlock will be disabled.\n"
        "  Run: pip install scipy",
        RuntimeWarning, stacklevel=1,
    )

# ── YOLOv8 — import deferred until background thread to prevent PyTorch/CUDA
# ── from running at import time and blocking Uvicorn's startup.
# ── Call _load_yolo() once from inside run_ai_pipeline() before any use.
YOLO = None          # populated by _load_yolo()
_YOLO_AVAILABLE = False

def _load_yolo():
    """
    Import ultralytics + PyTorch exactly ONCE, from the AI background thread.
    Never call this at module scope — PyTorch probes CUDA drivers on import
    and can silently hang for 15-60 s on certain Windows/driver combinations.
    """
    global YOLO, _YOLO_AVAILABLE
    if _YOLO_AVAILABLE:
        return True   # already loaded
    print("[YOLO] Importing ultralytics (PyTorch init — may take a few seconds)...")
    try:
        from ultralytics import YOLO as _Y
        YOLO = _Y
        _YOLO_AVAILABLE = True
        print("[YOLO] ultralytics imported OK.")
        return True
    except ImportError:
        _YOLO_AVAILABLE = False
        warnings.warn(
            "ultralytics not installed — face/vehicle detection disabled.\n"
            "  Run: pip install ultralytics",
            RuntimeWarning, stacklevel=2,
        )
        return False
    except Exception as exc:
        _YOLO_AVAILABLE = False
        print(f"[YOLO] Import failed: {exc}")
        return False

ESP32_IP = "10.212.43.45"
PRESENCE_URL = f"http://{ESP32_IP}/check-presence"
GATE_URL = f"http://{ESP32_IP}/open-gate"

# Database: Plate -> (Color, Type)
AUTHORIZED_DB = {
    "TN01AB1234": ("White", "SUV"),
    "KA05XY9876": ("Red", "Sedan")
}

# ============================================================
# PRIVACY FILTER ENGINE — GDPR-Compliant Anonymisation
# ============================================================
class PrivacyFilter:
    """
    Singleton that loads ML models ONCE and re-uses them every frame.

    Face anonymisation:
        Primary  → YOLOv8-face (ultralytics) — deep-learning, high recall
        Fallback → OpenCV Haar cascade       — zero extra dependencies

    Plate anonymisation:
        Stage 1  → OpenCV Haar plate cascade   — fast, low false-negative rate
        Stage 2  → Contour + aspect-ratio gate — supplements / replaces Stage 1
        Effect   → Mosaic pixelation (harder to reverse than Gaussian blur)

    GDPR note: even the primary (authorised) plate is pixelated in the
    saved image by default — its text was already captured by OCR and
    stored in the CSV log, so the raw image no longer needs to expose it.
    """

    _instance  = None
    _init_lock = threading.Lock()

    # ── Singleton constructor ─────────────────────────────────────────────
    def __new__(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._ready      = False
                    obj._yolo_faces = False
                    cls._instance   = obj
        return cls._instance

    # ── Lazy model bootstrap (called on first inference, not at import) ───
    def _bootstrap(self):
        if self._ready:
            return

        # ── Ensure PyTorch / ultralytics is loaded (deferred from import time) ──
        _load_yolo()   # no-op if already done; sets _YOLO_AVAILABLE + YOLO global

        # ── Face detector ─────────────────────────────────────────────────
        if _YOLO_AVAILABLE:
            try:
                # yolov8n-face.pt auto-downloads from Ultralytics Hub (~6 MB).
                # For air-gapped deployments, place the file beside server.py
                # and replace the string with its absolute path.
                self._face_model = YOLO("yolov8n-face.pt")
                self._yolo_faces = True
                print("[PRIVACY] YOLOv8-face model loaded.")
            except Exception as exc:
                print(f"[PRIVACY] YOLOv8-face load failed ({exc}). Using Haar fallback.")

        if not self._yolo_faces:
            self._face_haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            print("[PRIVACY] Haar cascade face detector loaded (fallback).")

        # ── Background plate detector ──────────────────────────────────────
        self._plate_haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
        )
        if self._plate_haar.empty():
            print("[PRIVACY] Plate Haar cascade not found — contour-only mode.")

        self._ready = True

    # ── Geometry helpers ─────────────────────────────────────────────────
    @staticmethod
    def _pad_box(x, y, w, h, pad, shape):
        """Expand a bbox by `pad` pixels, clamped to frame boundaries."""
        H, W = shape[:2]
        return (
            max(0, x - pad),
            max(0, y - pad),
            min(W, x + w + pad) - max(0, x - pad),
            min(H, y + h + pad) - max(0, y - pad),
        )

    # ── Per-ROI anonymisation primitives ────────────────────────────────
    @staticmethod
    def _blur_roi(frame, x, y, w, h, ksize=99):
        """Strong Gaussian blur — used for faces."""
        k = ksize if ksize % 2 == 1 else ksize + 1
        frame[y:y+h, x:x+w] = cv2.GaussianBlur(frame[y:y+h, x:x+w], (k, k), 0)

    @staticmethod
    def _pixelate_roi(frame, x, y, w, h, mosaic=10):
        """
        Mosaic pixelation — used for plates.
        Downscale to `mosaic`×`mosaic` then nearest-neighbour upscale.
        Much harder to reverse-engineer than Gaussian blur.
        """
        if w < 1 or h < 1:
            return
        roi   = frame[y:y+h, x:x+w]
        tiny  = cv2.resize(roi,  (mosaic, mosaic), interpolation=cv2.INTER_LINEAR)
        frame[y:y+h, x:x+w] = cv2.resize(tiny, (w, h), interpolation=cv2.INTER_NEAREST)

    # ── Detectors ────────────────────────────────────────────────────────
    def _detect_faces(self, frame):
        """
        Returns [(x,y,w,h), ...] for all detected faces.
        Inference runs on a 640px-wide thumbnail to cap latency;
        boxes are scaled back to the original resolution.
        """
        h0, w0 = frame.shape[:2]
        scale  = min(1.0, 640 / w0)
        small  = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)))
        boxes  = []

        if self._yolo_faces:
            results = self._face_model.predict(
                small, imgsz=640, conf=0.45, iou=0.45, verbose=False
            )
            for r in results:
                for box in r.boxes.xyxy.tolist():
                    x1, y1, x2, y2 = (int(v / scale) for v in box[:4])
                    boxes.append((x1, y1, x2 - x1, y2 - y1))
        else:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            dets = self._face_haar.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
            )
            for (fx, fy, fw, fh) in (dets if len(dets) > 0 else []):
                boxes.append((
                    int(fx / scale), int(fy / scale),
                    int(fw / scale), int(fh / scale),
                ))
        return boxes

    def _detect_plates(self, frame):
        """
        Returns [(x,y,w,h), ...] for all visible license plates.

        Two complementary stages:
          Stage 1 – Haar cascade   : high speed, catches well-aligned plates.
          Stage 2 – Contour filter : catches plates the cascade misses;
                                     uses aspect ratio (2.0 – 6.5) and area
                                     constraints typical of real plates.
        Duplicate regions are harmless (overlapping pixelation is idempotent).
        """
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = []

        # Strict Stage 1 — Haar cascade with confidence scores
        if not self._plate_haar.empty():
            dets, _, weights = self._plate_haar.detectMultiScale3(
                gray, scaleFactor=1.1, minNeighbors=4,
                minSize=(60, 20), flags=cv2.CASCADE_SCALE_IMAGE, outputRejectLevels=True
            )
            for i in range(len(dets)):
                if float(weights[i]) > 0.5: # Only anonymize objects with confidence > 0.5 to reduce ghosts
                    boxes.append(tuple(dets[i]))

        # Stage 2 (Contours) is intentionally bypassed to eliminate 0-confidence ghost detections
        return boxes

    # ── Public API ───────────────────────────────────────────────────────
    def process(self, frame, primary_plate_text=None):
        """
        Return a fully anonymised COPY of `frame`.

        - All faces    → strong Gaussian blur (σ auto, kernel 99×99)
        - All plates   → mosaic pixelation (10×10 mosaic)

        `primary_plate_text` is accepted for API compatibility and future
        spatial-exclusion logic but is intentionally NOT used to skip the
        primary plate: even the authorised plate should be anonymised in
        stored images (GDPR Art. 5 — data minimisation).
        """
        self._bootstrap()
        out = frame.copy()  # Never mutate the caller's frame

        # 1. Anonymise faces
        face_boxes = self._detect_faces(out)
        for (x, y, w, h) in face_boxes:
            x, y, w, h = self._pad_box(x, y, w, h, 20, out.shape)  # expand bbox
            self._blur_roi(out, x, y, w, h)

        # 2. Pixelate background plates
        plate_boxes = self._detect_plates(out)
        for (x, y, w, h) in plate_boxes:
            self._pixelate_roi(out, x, y, w, h)

        print(f"[PRIVACY] Anonymised {len(face_boxes)} face(s), "
              f"{len(plate_boxes)} plate(s).")
        return out


# ── Null global — instantiated lazily inside apply_privacy_blur() ────────────
_privacy_filter = None

# ============================================================
# LIFI RECEIVER ENGINE — Flashlight Pattern Unlock
# ============================================================
class LiFiReceiver:
    """
    Tracks brightness over a 3-second rolling window and applies
    Fast Fourier Transform (FFT) to detect specific stroboscopic
    flashlight frequencies (e.g. 3 Hz) to act as a fallback unlock.
    """
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    # 3-second rolling window at ~15-20 fps = ~60 frames
                    obj.history = collections.deque(maxlen=60)
                    obj.target_freq = 3.0 # Target: 3 flashes per second
                    obj.freq_tolerance = 0.8 # Allow 2.2 Hz - 3.8 Hz due to manual flashing variations
                    cls._instance = obj
        return cls._instance

    def process(self, frame):
        if not _SCIPY_AVAILABLE:
            return False

        # Extract maximum brightness (representing a direct flashlight beam)
        # We blur slightly to reduce stray pixel noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        _, max_val, _, _ = cv2.minMaxLoc(gray_blurred)
        
        now = time.time()
        self.history.append((now, max_val))
        
        # Need at least 2 seconds of history to compute meaningful frequencies
        if len(self.history) < 20: 
            return False
            
        times = np.array([t for t, v in self.history])
        vals = np.array([v for t, v in self.history])
        
        duration = times[-1] - times[0]
        if duration < 2.0:
            return False
            
        fps = len(self.history) / duration
        
        # Detrend the signal (remove DC component & slow ambient lighting shifts)
        vals_detrended = scipy.signal.detrend(vals)
        
        # Require a minimum variance to ignore entirely static scenes
        if np.std(vals_detrended) < 15:
            return False

        # Apply Real Fast Fourier Transform (rfft) since image brightness is real
        N = len(vals_detrended)
        T = 1.0 / fps
        yf = rfft(vals_detrended)
        xf = rfftfreq(N, T)
        
        magnitudes = np.abs(yf)
        
        if len(magnitudes) == 0:
            return False
            
        # Ignore the 0Hz DC bin just in case detrend wasn't perfect, find dominant frequency
        peak_idx = np.argmax(magnitudes[1:]) + 1 
        peak_freq = xf[peak_idx]
        peak_mag = magnitudes[peak_idx]
        
        # Match frequency within tolerance and ensure a strong signal magnitude
        if peak_mag > 100 and abs(peak_freq - self.target_freq) <= self.freq_tolerance:
            print(f"[LIFI] 🔦 Signal detected! Freq: {peak_freq:.2f} Hz, Mag: {peak_mag:.2f}")
            self.history.clear() # Reset window to prevent continuous triggers
            return True
            
        return False

# ── Null global — instantiated lazily inside check_lifi_signal() ────────────
_lifi_receiver = None

# ============================================================
# SHARED DASHBOARD STATE  (read by dashboard.py via import)
# ============================================================
import base64

dashboard_state = {
    "intensity":    0.0,
    "lifi_state":   False,
    "pulse_count":  0,
    "gate_status":  "IDLE",       # "IDLE" | "OPEN" | "DENIED"
    "plate_text":   "",
    "authorized":   False,
    "last_gate_ts": "",
}
# Raw JPEG bytes of the latest annotated frame — updated every loop tick
shared_frame_jpg: bytes = b""
# Thread-safety lock for the shared frame
_frame_lock = threading.Lock()

def _push_frame(frame):
    """JPEG-encode `frame` and store in the shared buffer."""
    global shared_frame_jpg
    ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ret:
        with _frame_lock:
            shared_frame_jpg = buf.tobytes()

def _emit(sio, event, data):
    """Safe emit — no-ops when running without a dashboard."""
    if sio is not None:
        try:
            sio.emit(event, data)
        except Exception:
            pass

def _log(sio, message, level="info"):
    """Emit a log line AND print to terminal."""
    print(message)
    _emit(sio, "log_line", {"message": message, "level": level})

def open_gate(sio=None):
    global dashboard_state
    try:
        requests.get(GATE_URL, timeout=3)
        _log(sio, "[GATE] Gate opened.", "success")
    except Exception:
        pass
    dashboard_state["gate_status"] = "OPEN"
    dashboard_state["last_gate_ts"] = datetime.datetime.now().strftime("%H:%M:%S")
    _emit(sio, "gate_event", {
        "status":    "OPEN",
        "plate":     dashboard_state["plate_text"],
        "timestamp": dashboard_state["last_gate_ts"],
    })

# ==========================================
# FEATURE 3: Privacy Anonymization (GDPR)
# ==========================================
def apply_privacy_blur(frame, primary_plate_text=None):
    """
    GDPR-compliant frame anonymisation.

    Replaces the legacy single-model Haar blur with:
      • YOLOv8-face deep detector  → Gaussian blur  on ALL detected faces
      • Two-stage plate detector   → Mosaic pixelation on ALL visible plates

    Args:
        frame:               BGR frame captured from the ESP32-CAM.
        primary_plate_text:  OCR'd plate string of the authorised vehicle.
                             Passed through to PrivacyFilter for future
                             spatial-exclusion support.
    Returns:
        A new anonymised frame; the original `frame` is not modified.
    """
    global _privacy_filter
    if _privacy_filter is None:           # lazy-init on first call
        _privacy_filter = PrivacyFilter()
    return _privacy_filter.process(frame, primary_plate_text=primary_plate_text)

# ============================================================
# VISUAL AUTHENTICATOR ENGINE — Multi-Factor Plate Cloning Prevention
# ============================================================
class VehicleAuthenticator:
    """
    Singleton class to manage the YOLOv8 object detection model 
    and HSV color extraction. Loaded lazily.
    """
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._ready = False
                    cls._instance = obj
        return cls._instance

    def _bootstrap(self):
        if self._ready:
            return
        # ── Ensure PyTorch / ultralytics is loaded (deferred from import time) ──
        _load_yolo()   # no-op if already done; sets _YOLO_AVAILABLE + YOLO global
        if _YOLO_AVAILABLE:
            try:
                # yolov8n.pt is the standard object detection model (~6.2 MB)
                self._object_model = YOLO("yolov8n.pt")
                print("[AUTH] YOLOv8n object detection model loaded.")
            except Exception as exc:
                print(f"[AUTH] YOLOv8n load failed ({exc}).")
                self._object_model = None
        else:
            self._object_model = None
            print("[AUTH] Warning: ultralytics not installed, visual auth disabled.")
        self._ready = True

    def _get_dominant_color(self, roi):
        """Extracts the dominant color using HSV thresholding."""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define color boundaries in HSV
        colors = {
            "Red": [
                (np.array([0, 70, 50]), np.array([10, 255, 255])),
                (np.array([170, 70, 50]), np.array([180, 255, 255]))
            ],
            "Blue": [(np.array([100, 150, 0]), np.array([140, 255, 255]))],
            "Green": [(np.array([35, 50, 50]), np.array([85, 255, 255]))],
            "Yellow": [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
            # White: Low saturation, high value
            "White": [(np.array([0, 0, 200]), np.array([180, 30, 255]))],
            # Black: Any hue, any saturation, low value
            "Black": [(np.array([0, 0, 0]), np.array([180, 255, 50]))],
            # Silver/Grey: Low saturation, medium value
            "Silver": [(np.array([0, 0, 50]), np.array([180, 30, 200]))],
        }

        best_color = None
        max_pixels = 0
        
        for color_name, ranges in colors.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for (lower, upper) in ranges:
                mask |= cv2.inRange(hsv, lower, upper)
            
            pixel_count = cv2.countNonZero(mask)
            if pixel_count > max_pixels:
                max_pixels = pixel_count
                best_color = color_name
                
        return best_color

    def verify(self, frame, expected_color, expected_type):
        """Runs the 2-factor verification on the frame."""
        self._bootstrap()
        
        if not self._object_model:
            print("[AUTH] Skipping multi-factor auth (model not loaded).")
            return True # Fail open

        results = self._object_model.predict(frame, conf=0.4, verbose=False)
        
        # Map DB expected_type to YOLO COCO classes 
        # COCO: 2=car, 3=motorcycle, 5=bus, 7=truck
        yolo_classes = []
        expected_type_lower = expected_type.lower()
        if expected_type_lower in ["sedan", "suv", "hatchback", "car"]:
            yolo_classes.append(2)
        elif expected_type_lower in ["truck", "pickup"]:
            yolo_classes.append(7)
        elif expected_type_lower in ["motorcycle", "bike"]:
            yolo_classes.append(3)
        
        best_box = None
        max_area = 0
        detected_cls = -1
        
        # 1. Verify Structure (Vehicle Type)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id in [2, 3, 5, 7]: # It's a vehicle
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    area = (x2-x1) * (y2-y1)
                    if area > max_area:
                        max_area = area
                        best_box = (x1, y1, x2, y2)
                        detected_cls = cls_id

        if not best_box:
            print("[AUTH] ❌ No vehicle detected by YOLO.")
            return False

        # If a specific type was requested and mapped, verify it
        if yolo_classes and detected_cls not in yolo_classes:
            print(f"[AUTH] ❌ Type Mismatch: Expected {expected_type}, got YOLO cls {detected_cls}")
            return False

        # 2. Verify Color
        x1, y1, x2, y2 = best_box
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w//2, y1 + h//2
        # Sample pixels from the top 20% of the car's detection box (where the actual paint is)
        crop_w, crop_h = int(w * 0.6), int(h * 0.2)
        crop_x1 = max(0, cx - crop_w // 2)
        crop_y1 = max(0, y1 + int(h * 0.05)) # Top 5% to 25% down from roof
        crop_x2 = min(frame.shape[1], crop_x1 + crop_w)
        crop_y2 = min(frame.shape[0], crop_y1 + crop_h)

        roi = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if roi.size == 0:
            print("[AUTH] ❌ Invalid ROI for color.")
            return False

        detected_color = self._get_dominant_color(roi)
        
        if not detected_color:
            print("[AUTH] ❌ Unknown color.")
            return False
            
        if detected_color.lower() != expected_color.lower():
            if expected_color.title() == "White" and detected_color.title() in ["Gray", "Silver", "Black"]:
                print(f"[AUTH] ⚠️ Partial Match: Expected White, detected {detected_color.title()}. Allowing for demo.")
                return True
            print(f"[AUTH] ❌ Color Mismatch: Expected {expected_color}, but detected {detected_color}")
            return False

        print(f"[AUTH] ✅ Visual Auth Passed: {detected_color} {expected_type}")
        return True

# ── Null global — instantiated lazily inside verify_vehicle_attributes() ─────
_vehicle_authenticator = None


# ==========================================
# FEATURE 4: Multi-Factor Visual Auth
# ==========================================
def verify_vehicle_attributes(frame, detected_plate):
    """
    Validates if the detected plate matches the physical car color/type 
    to prevent printed-plate cloning.
    """
    global _vehicle_authenticator
    if _vehicle_authenticator is None:    # lazy-init on first call
        _vehicle_authenticator = VehicleAuthenticator()
    expected_color, expected_type = AUTHORIZED_DB.get(detected_plate, (None, None))
    if not expected_color:
        return False
    return _vehicle_authenticator.verify(frame, expected_color, expected_type)

# ==========================================
# FEATURE 5: Predictive Analytics Logging
# ==========================================
def log_entry():
    """Logs timestamp for Time-Series Regression."""
    with open('parking_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.datetime.now(), 1]) # 1 = Entry

# ==========================================
# FEATURE 6: Li-Fi (Flashlight) Unlock
# ==========================================
TEST_MODE = False
def check_lifi_signal(frame):
    """Returns the maximum brightness intensity of the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, max_val, _, _ = cv2.minMaxLoc(gray_blurred)
    return max_val

def _open_camera_with_timeout(index=0, use_dshow=True, timeout_sec=10):
    """
    Opens cv2.VideoCapture in a separate thread so a blocked DirectShow
    enumeration cannot hang the whole AI worker indefinitely.

    Returns the opened VideoCapture on success, or None on timeout/failure.
    """
    result = [None]
    event  = threading.Event()

    def _open():
        backend = cv2.CAP_DSHOW if use_dshow else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        result[0] = cap
        event.set()

    t = threading.Thread(target=_open, daemon=True)
    t.start()
    if not event.wait(timeout=timeout_sec):
        print(f"[CAMERA] ✗ Timed out after {timeout_sec}s waiting for VideoCapture({index}). "
              f"Check that no other app is holding the camera.")
        return None

    cap = result[0]
    if cap is None or not cap.isOpened():
        print(f"[CAMERA] ✗ VideoCapture({index}) opened but is not ready.")
        return None

    return cap


def run_ai_pipeline(sio=None):
    """Core AI loop. Pass a Socket.IO instance to enable real-time dashboard."""
    global dashboard_state

    # ── Boot step A: import ultralytics + PyTorch (deferred — never at module scope) ──
    # _load_yolo() sets the global YOLO symbol and _YOLO_AVAILABLE flag used by
    # PrivacyFilter and VehicleAuthenticator. Safe to call multiple times (no-op).
    print("[BOOT] AI Pipeline — Step A: Loading YOLO / PyTorch (may take 5-15 s)...")
    _load_yolo()
    print("[BOOT] AI Pipeline — Step A: YOLO ready." if _YOLO_AVAILABLE
          else "[BOOT] AI Pipeline — Step A: YOLO unavailable, using Haar fallback.")

    # ── Boot step B: open the webcam with a strict OS-level timeout ───────────
    # cv2.VideoCapture with CAP_DSHOW can silently hang 15-60 s on Windows while
    # it enumerates all DirectShow devices. We run it in a daemon thread and
    # enforce a hard 10-second deadline via threading.Event.
    print("[BOOT] AI Pipeline — Step B: Opening webcam (10 s timeout)...")
    cap = _open_camera_with_timeout(index=1, use_dshow=True, timeout_sec=10)

    if cap is None:
        print("[BOOT] AI Pipeline — Step B: index 1 failed, retrying index 0 (no DSHOW)...")
        cap = _open_camera_with_timeout(index=0, use_dshow=False, timeout_sec=10)

    if cap is None or not cap.isOpened():
        _log(sio, "[ERROR] Could not open any webcam. AI pipeline aborted.", "error")
        return

    print("[BOOT] AI Pipeline — Step B: Webcam opened successfully. Pipeline running.")
        
    _log(sio, "[SERVER] System running. Waiting for vehicle trigger from ESP32...", "info")
    simple_light_start = 0
    lifi_pulse_count = 0
    lifi_last_state = False
    last_pulse_timestamp = 0

    try:
        while True:
            try:
                # 1. Ask ESP32 if a car is there
                resp = requests.get(PRESENCE_URL, headers={'Connection': 'close'}, timeout=5)
                
                # CASE: NO VEHICLE
                if 'EMPTY' in resp.text: 
                    if TEST_MODE:
                        frame = cv2.imread("test_car.jpg")
                        ret = True if frame is not None else False
                    else:
                        ret, frame = cap.read()
                        
                    if ret:
                        intensity_value = check_lifi_signal(frame)
                        LIFI_THRESHOLD = 200
                        
                        # Step A: Determine current_state based explicitly on threshold
                        current_state = True if intensity_value > LIFI_THRESHOLD else False
                        
                        # Sync States: Visual indicator uses current_state
                        color = (0, 255, 0) if current_state else (0, 0, 255)
                        cv2.putText(frame, f"LiFi: {intensity_value:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Step B: Edge Detection Logic
                        if current_state and not lifi_last_state:
                            lifi_pulse_count += 1
                            last_pulse_timestamp = time.time()

                        log_msg = f"[LIFI] Intensity: {intensity_value:.1f} | State: {current_state} | Pulses: {lifi_pulse_count}"
                        _log(sio, log_msg, "lifi")
                        
                        # Auto-Reset
                        if time.time() - last_pulse_timestamp > 3.0:
                            lifi_pulse_count = 0
                            
                        # Step C: Update last state
                        lifi_last_state = current_state
                        
                        # Update shared dashboard state
                        dashboard_state["intensity"]   = float(intensity_value)
                        dashboard_state["lifi_state"]  = current_state
                        dashboard_state["pulse_count"] = lifi_pulse_count
                        _emit(sio, "lifi_update", {
                            "intensity":   dashboard_state["intensity"],
                            "state":       current_state,
                            "pulse_count": lifi_pulse_count,
                        })

                        # Visual Feedback (OpenCV window — kept as fallback)
                        for i in range(lifi_pulse_count):
                            cv2.circle(frame, (30 + i * 40, 70), 10, (0, 255, 0), -1)

                        _push_frame(frame)
                            
                        if lifi_pulse_count >= 3:
                            _log(sio, "[LIFI] SUCCESS: 3 Pulses Detected!", "success")
                            dashboard_state["plate_text"] = "Li-Fi Override"
                            open_gate(sio)
                            lifi_pulse_count = 0
                            simple_light_start = 0
                            time.sleep(5)
                        elif intensity_value > LIFI_THRESHOLD:
                            if simple_light_start == 0:
                                simple_light_start = time.time()
                            elif time.time() - simple_light_start > 2.0:
                                _log(sio, "[LIFI] Simple Light Mode Triggered!", "success")
                                dashboard_state["plate_text"] = "Li-Fi Override"
                                open_gate(sio)
                                simple_light_start = 0
                                time.sleep(5)
                        else:
                            simple_light_start = 0

                        cv2.imshow("Smart LPR System", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): 
                        break
                    time.sleep(0.5)
                    continue
                    
                # CASE: VEHICLE DETECTED
                if 'DETECTED' in resp.text:
                    plate_text = ""
                    print("[SERVER] Trigger received! Snapping frame...")
                    if TEST_MODE:
                        frame = cv2.imread("test_car.jpg")
                        ret = True if frame is not None else False
                    else:
                        ret, frame = cap.read()
                        
                    if not ret:
                        continue

                    # --- OVERLAY ---
                    intensity_value = check_lifi_signal(frame)
                    LIFI_THRESHOLD = 200
                    
                    # Step A: Determine current_state based explicitly on threshold
                    current_state = True if intensity_value > LIFI_THRESHOLD else False
                    
                    # Sync States: Visual indicator uses current_state
                    color = (0, 255, 0) if current_state else (0, 0, 255)
                    cv2.putText(frame, f"LiFi: {intensity_value:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Step B: Edge Detection Logic
                    if current_state and not lifi_last_state:
                        lifi_pulse_count += 1
                        last_pulse_timestamp = time.time()

                    _log(sio, f"[LIFI] Intensity: {intensity_value:.1f} | State: {current_state} | Pulses: {lifi_pulse_count}", "lifi")
                    
                    # Auto-Reset
                    if time.time() - last_pulse_timestamp > 3.0:
                        lifi_pulse_count = 0
                        
                    # Step C: Update last state
                    lifi_last_state = current_state
                    
                    # Update shared dashboard state
                    dashboard_state["intensity"]   = float(intensity_value)
                    dashboard_state["lifi_state"]  = current_state
                    dashboard_state["pulse_count"] = lifi_pulse_count
                    _emit(sio, "lifi_update", {
                        "intensity":   dashboard_state["intensity"],
                        "state":       current_state,
                        "pulse_count": lifi_pulse_count,
                    })

                    # Visual Feedback
                    for i in range(lifi_pulse_count):
                        cv2.circle(frame, (30 + i * 40, 70), 10, (0, 255, 0), -1)

                    # --- STEP 1: LI-FI OVERRIDE ---
                    if lifi_pulse_count >= 3 or (intensity_value > LIFI_THRESHOLD and simple_light_start and time.time() - simple_light_start > 2.0):
                        _log(sio, "[LIFI] SUCCESS: 3 Pulses Detected! Bypassing OCR...", "success")
                        dashboard_state["plate_text"] = "Li-Fi Override"
                        open_gate(sio)
                        lifi_pulse_count = 0
                        simple_light_start = 0
                        _push_frame(frame)
                        time.sleep(5)
                        continue

                    # --- STEP 2: OCR ---
                    h, w = frame.shape[:2]
                    cropped_frame = frame[int(h*0.45):int(h*0.65), int(w*0.35):int(w*0.65)]
                    cv2.imshow('OCR Focus', cropped_frame)
                    
                    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
                    # Upscale by 2x for better Tesseract accuracy
                    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    # Apply Otsu Thresholding to make text pure black/white
                    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    
                    # Downscale for Tesseract if image is too large
                    if gray.shape[1] > 1000:
                        gray = cv2.resize(gray, (0,0), fx=0.5, fy=0.5)
                    
                    # Save the 'Brain' Image
                    cv2.imwrite('processed_ocr.jpg', gray)
                    
                    raw_ocr_result = pytesseract.image_to_string(gray, config='--psm 7').strip()
                    print(f"[DEBUG] Raw OCR: '{raw_ocr_result}'")
                    
                    # Strip all non-alphanumeric characters and force uppercase
                    plate_text = "".join(c for c in raw_ocr_result if c.isalnum()).upper()
                    
                    # Character normalization for common OCR mistakes
                    plate_text = plate_text.replace('O', '0').replace('I', '1').replace('L', '1')
                    print(f"[DEBUG] Cleaned Plate: '{plate_text}'")

                    if not plate_text or len(plate_text) <= 3:
                        _log(sio, f"[DEBUG] FAILED: OCR returned '{plate_text}' — too short.", "error")
                        _push_frame(frame)
                        cv2.imshow("Smart LPR System", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        continue

                    # --- STEP 3: AUTHORIZATION ---
                    dashboard_state["plate_text"] = plate_text
                    _emit(sio, "ocr_result", {"plate_text": plate_text, "authorized": plate_text in AUTHORIZED_DB})

                    if plate_text in AUTHORIZED_DB:
                        _log(sio, f"[SUCCESS] Authorized vehicle: {plate_text}", "success")
                        dashboard_state["authorized"] = True
                        open_gate(sio)
                        
                        # --- STEP 4: MULTI-FACTOR VISUAL AUTH ---
                        verify_vehicle_attributes(frame, plate_text)
                            
                        # --- STEP 5: PRIVACY & LOGGING ---
                        safe_frame = apply_privacy_blur(frame, primary_plate_text=plate_text)
                        cv2.imwrite(f"logs/{plate_text}_{time.time()}.jpg", safe_frame)
                        log_entry()
                        
                        time.sleep(5)
                    else:
                        _log(sio, f"[DENIED] Plate '{plate_text}' not in database.", "error")
                        dashboard_state["authorized"] = False
                        dashboard_state["gate_status"] = "DENIED"
                        _emit(sio, "gate_event", {"status": "DENIED", "plate": plate_text,
                                                  "timestamp": datetime.datetime.now().strftime("%H:%M:%S")})

                    _push_frame(frame)
                    cv2.imshow("Smart LPR System", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
            except Exception as e:
                _log(sio, f"[CONN ERROR] {e}", "error")
                time.sleep(1)
                
    except KeyboardInterrupt:
        _log(sio, "\n[SERVER] Shutting down cleanly...", "info")

    cap.release()
    cv2.destroyAllWindows()

# Legacy entry point — runs without dashboard
def main():
    run_ai_pipeline(sio=None)

if __name__ == "__main__":
    main()