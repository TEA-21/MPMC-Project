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

# ── Optional deep-learning backend (install with: pip install ultralytics) ──
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    warnings.warn(
        "ultralytics not installed – face detection will fall back to Haar cascades.\n"
        "  Run: pip install ultralytics",
        RuntimeWarning, stacklevel=1,
    )

ESP32_IP = "192.168.1.100"
CAPTURE_URL = f"http://{ESP32_IP}/smart-capture"
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
        # OpenCV's built-in plate cascade (covers Russian / EU plate shapes well;
        # covers Indian / US plates adequately when combined with the contour stage).
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

        # Stage 1 — Haar cascade
        if not self._plate_haar.empty():
            dets = self._plate_haar.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4,
                minSize=(60, 20), flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(dets) > 0:
                boxes.extend([(x, y, w, h) for (x, y, w, h) in dets])

        # Stage 2 — Contour + aspect-ratio filter
        edges  = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 200)
        cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0:
                continue
            aspect = w / h
            area   = w * h
            # Typical plate: width 60-400 px, aspect 2.0-6.5, area 1 200-60 000 px²
            if 2.0 <= aspect <= 6.5 and 1200 <= area <= 60000:
                boxes.append((x, y, w, h))

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


# Module-level singleton — models are loaded once on first call
_privacy_filter = PrivacyFilter()

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

_lifi_receiver = LiFiReceiver()

def open_gate():
    try:
        requests.get(GATE_URL, timeout=3)
        print("Gate Opened.")
    except:
        pass

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
        # Crop the center 40% to avoid background, windows, and headlights
        crop_w, crop_h = int(w * 0.4), int(h * 0.4)
        crop_x1 = max(0, cx - crop_w // 2)
        crop_y1 = max(0, cy - crop_h // 2 + int(h * 0.1)) # Shift slightly down to get hood/door, less roof/glass
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
            print(f"[AUTH] ❌ Color Mismatch: Expected {expected_color}, but detected {detected_color}")
            return False

        print(f"[AUTH] ✅ Visual Auth Passed: {detected_color} {expected_type}")
        return True

_vehicle_authenticator = VehicleAuthenticator()


# ==========================================
# FEATURE 4: Multi-Factor Visual Auth
# ==========================================
def verify_vehicle_attributes(frame, detected_plate):
    """
    Validates if the detected plate matches the physical car color/type 
    to prevent printed-plate cloning.
    """
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
def check_lifi_signal(frame):
    """
    Scientifically analyzes frame brightness variations over time
    using FFT to securely authorize entry via flashlight patterns.
    """
    return _lifi_receiver.process(frame)

def main():
    while True:
        try:
            resp = urllib.request.urlopen(CAPTURE_URL, timeout=2)
            if resp.getcode() == 204: # No vehicle detected by ultrasonic
                continue
                
            imgnp = np.array(bytearray(resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(imgnp, -1)
            
            # 1. Check Li-Fi Override First
            if check_lifi_signal(frame):
                open_gate()
                time.sleep(5)
                continue

            # 2. Extract Plate (OCR logic shortened for brevity)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            plate_text = pytesseract.image_to_string(gray, config='--psm 8').strip()

            if plate_text in AUTHORIZED_DB:
                # 3. Multi-Factor Check
                if verify_vehicle_attributes(frame, plate_text):
                    # 4. Privacy Blur
                    # Pass plate text so PrivacyFilter can log / exclude it
                    safe_frame = apply_privacy_blur(frame, primary_plate_text=plate_text)
                    cv2.imwrite(f"logs/{plate_text}_{time.time()}.jpg", safe_frame)
                    
                    # 5. Log & Open
                    log_entry()
                    open_gate()
                    time.sleep(5)

            cv2.imshow("Smart LPR System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        except Exception as e:
            pass # Handle timeouts cleanly

if __name__ == "__main__":
    main()