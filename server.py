# =============================================================================
# LPR Parking Gate — Backend Processing Server
# =============================================================================
# Target Runtime  : Python 3.10+
# Author          : TEA-21 / MPMC Project
# Description     : Central AI processing engine for the single-lane LPR gate
#                   system.  Polls the ESP32-CAM edge device, runs multi-gate
#                   vehicle authentication, performs GDPR-compliant anonymisation,
#                   and triggers the servo barrier via HTTP.
#
# Full Processing Pipeline (dual-threaded):
#
#   HTTP Poller Thread  ──┬── Gate 3 : OCR  Identity Resolution
#   (every 2.0 s)         └── Gate 4 : Anti-Spoof  (YOLO + HSV colour)
#
#   VLC/Li-Fi Thread    ──── FFT Frequency Analysis → /open-gate  (bypasses OCR)
#
# Required packages:
#   pip install opencv-python pytesseract ultralytics scipy numpy pandas requests
#
# Tesseract binary:
#   Windows : https://github.com/UB-Mannheim/tesseract/wiki
#             Default: C:\Program Files\Tesseract-OCR\tesseract.exe
#   Linux   : sudo apt install tesseract-ocr
# =============================================================================

import cv2
import numpy as np
import pandas as pd          # available for future analytics extensions
import requests
import threading
import time
import datetime
import collections
import os
import re
import csv
import difflib
import logging

import pytesseract

import scipy.signal
from scipy.fft import rfft, rfftfreq

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tesseract Binary Path  (Windows requires explicit path)
# ---------------------------------------------------------------------------
# Set environment variable TESSERACT_CMD to override, e.g.:
#   set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
TESSERACT_CMD = os.environ.get(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)
if os.path.isfile(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# ---------------------------------------------------------------------------
# Network / Hardware Configuration
# ---------------------------------------------------------------------------
# The ESP32 IP is printed on Arduino Serial Monitor after each boot.
# It is assigned by the mobile hotspot DHCP and may change each session.
# Override at runtime:  set ESP32_IP=192.168.x.x
ESP32_IP          = os.environ.get("ESP32_IP", "10.164.250.54")
SMART_CAPTURE_URL = f"http://{ESP32_IP}/smart-capture"  # polled every 2.0 s
OPEN_GATE_URL     = f"http://{ESP32_IP}/open-gate"      # GET to raise barrier

# Polling cadence (seconds)
POLL_INTERVAL_S = 2.0

# ---------------------------------------------------------------------------
# Authorised Vehicle Database  (in-memory dict, replace with DB in production)
# ---------------------------------------------------------------------------
# Format:  { "PLATE_TEXT": { "type": "<yolo_class>", "color": "<hsv_bin>" } }
# "type"  must be one of: car | motorcycle | bus | truck
# "color" must be one of: red | orange | yellow | green | blue | white | black | silver
AUTHORIZED_DB = {
    "TN01XX0001": {"type": "car",        "color": "white"},
    "TN01XX0002": {"type": "motorcycle", "color": "black"},
    "KA05AB1234": {"type": "truck",      "color": "silver"},
    "MH12CD5678": {"type": "bus",        "color": "blue"},
}

# ---------------------------------------------------------------------------
# YOLO Vehicle Class Mapping  (COCO dataset indices)
# ---------------------------------------------------------------------------
# Only these class IDs are considered vehicles for Gate 4 attribute verification.
# COCO: 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_YOLO_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# ---------------------------------------------------------------------------
# Seven Canonical HSV Colour Bins  (Gate 4 colour verification)
# ---------------------------------------------------------------------------
# Each entry: list of (lower_hsv, upper_hsv) NumPy arrays.
# Red wraps the HSV hue wheel and therefore requires two separate ranges.
HSV_COLOR_BINS = {
    "red":    [
                  (np.array([  0,  70,  50]), np.array([ 10, 255, 255])),
                  (np.array([170,  70,  50]), np.array([180, 255, 255])),
              ],
    "orange": [(np.array([ 11,  80,  80]), np.array([ 25, 255, 255]))],
    "yellow": [(np.array([ 26, 100, 100]), np.array([ 34, 255, 255]))],
    "green":  [(np.array([ 35,  50,  50]), np.array([ 85, 255, 255]))],
    "blue":   [(np.array([100, 100,  50]), np.array([140, 255, 255]))],
    "white":  [(np.array([  0,   0, 180]), np.array([180,  40, 255]))],
    "black":  [(np.array([  0,   0,   0]), np.array([180, 255,  60]))],
    "silver": [(np.array([  0,   0,  61]), np.array([180,  40, 179]))],
}

# ---------------------------------------------------------------------------
# Persistence Paths
# ---------------------------------------------------------------------------
LOG_CSV_PATH  = "parking_log.csv"
ANON_SAVE_DIR = "anonymised_frames"
os.makedirs(ANON_SAVE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# VLC / Li-Fi Signal Parameters
# ---------------------------------------------------------------------------
# The optical VLC transmitter (e.g. a smartphone strobe app) flashes at
# VLC_TARGET_FREQ_HZ.  A gate-open command is sent when:
#   |f_peak - TARGET_FREQ| <= FREQ_TOL  AND  |B[k_max]| > MIN_MAGNITUDE
VLC_TARGET_FREQ_HZ = 3.0    # Expected carrier frequency (Hz)
VLC_FREQ_TOL_HZ    = 0.8    # ±tolerance window (Hz)
VLC_MIN_MAGNITUDE  = 100.0  # Minimum FFT magnitude to accept as valid signal
VLC_WINDOW_FRAMES  = 45     # Rolling buffer depth (~3 s at 15 fps)

# ---------------------------------------------------------------------------
# Backend bridge — attributes expected by backend/main.py
# ---------------------------------------------------------------------------
# Shared MJPEG frame (bytes) updated by run_ai_pipeline; read by video_feed.
_frame_lock:      threading.Lock = threading.Lock()
shared_frame_jpg: bytes          = b""

# Live dashboard state dict — updated by run_ai_pipeline, read by telemetry.
dashboard_state: dict = {
    "intensity":   0.0,
    "lifi_state":  False,
    "pulse_count": 0,
    "gate_status": "IDLE",
    "plate_text":  "",
    "authorized":  False,
}

# Mirror of backend/main.py's LIFI_MODE_ACTIVE flag.
LIFI_MODE_ACTIVE: bool = False

# Emit shim — backend/main.py replaces this with a real Socket.IO emitter.
def _emit(sio_ignored, event: str, data: dict) -> None:  # noqa: E302
    """No-op emit; overridden by backend/main.py after import."""



# =============================================================================
#  THREAD-SAFE SINGLETON BASE
# =============================================================================
class _ThreadSafeSingleton:
    """
    Thread-safe double-checked locking singleton mixin.

    The first call to __new__() constructs the instance and stores it in
    cls._instance.  Subsequent calls return the same object.  The inner lock
    prevents a race where two threads simultaneously pass the outer None check
    before either acquires the lock (double-checked locking pattern).

    Subclasses should perform expensive initialisation (model loading) in a
    _bootstrap() method called lazily on first use, NOT in __new__ or __init__,
    to avoid blocking the import phase.
    """
    _instance  = None
    _init_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._init_lock:
                # Second check: another thread may have initialised while we waited
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._bootstrapped = False   # Subclasses set True after model load
                    cls._instance = obj
        return cls._instance


# =============================================================================
#  GATE 3 — OCR IDENTITY RESOLUTION ENGINE
# =============================================================================
class OCREngine(_ThreadSafeSingleton):
    """
    Singleton OCR engine wrapping Tesseract via pytesseract.

    Preprocessing pipeline applied before every recognition call:
      1. BGR → greyscale      : removes colour noise irrelevant to character edges.
      2. Gaussian blur (5×5)  : suppresses high-frequency JPEG artefacts that cause
                                Tesseract to misread compression blocks as strokes.
      3. Otsu thresholding    : adaptive binarisation — automatically selects the
                                optimal global threshold based on the frame histogram.
                                Works for both dark-text-on-white and white-text-on-
                                dark plates without manual parameter tuning.

    Tesseract PSM cascade:
      Pass 1 — ``--psm 8``  (single-word mode)
          Optimal for standard Indian single-line plates (e.g. MH12AB3456).
          Treats the OCR input as a single word / token without layout analysis.
      Pass 2 — ``--psm 6``  (block of text, fallback)
          Fallback for two-line two-wheeler plates whose state/RTO header row
          causes PSM 8 to concatenate multiple lines into garbage tokens.
          PSM 6 reads the plate in reading order and then normalisation strips
          whitespace, yielding the same alphanumeric string.

    Normalisation strips all non-alphanumeric characters and uppercases the
    result before AUTHORIZED_DB lookup.
    """

    # Whitelist: plates contain only capital letters and digits
    _CFG_PSM8 = (
        "--psm 8 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    _CFG_PSM6 = (
        "--psm 6 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    def _bootstrap(self) -> None:
        """No model weights to load; Tesseract is a system binary."""
        if not self._bootstrapped:
            log.info("[OCR] Tesseract engine ready (binary: %s).", TESSERACT_CMD)
            self._bootstrapped = True

    # ── Preprocessing ─────────────────────────────────────────────────────
    @staticmethod
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        """
        Return a binarised greyscale image ready for Tesseract.

        1. BGR → Grey   : colour carries no plate-text information.
        2. Gaussian blur : reduces JPEG block artefacts (5×5 kernel, σ=0 auto).
        3. Otsu thresh   : finds the bimodal histogram valley between plate
                           background and character pixels and binarises there.
        """
        grey    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grey, (5, 5), 0)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary

    @staticmethod
    def _normalise(raw: str) -> str:
        """Uppercase and strip everything that is not A-Z or 0-9."""
        return re.sub(r"[^A-Z0-9]", "", raw.upper())

    # ── Public API ─────────────────────────────────────────────────────────
    def recognise(self, frame: np.ndarray) -> str | None:
        """
        Run two-pass OCR on a full BGR frame and return the AUTHORIZED_DB key
        for the detected plate, or None if no authorised plate is found.

        Algorithm:
          1. Preprocess frame to binary.
          2. PSM 8  → normalise → fuzzy match against AUTHORIZED_DB keys.
          3. PSM 6  → normalise → fuzzy match (fallback for two-line plates).
          4. Return None if both passes fail to meet the similarity threshold.

        Fuzzy Matching — difflib.get_close_matches():
            Uses Python's built-in SequenceMatcher (Ratcliff/Obershelp algorithm)
            to compute a similarity ratio in [0, 1] between the OCR output and
            every key in AUTHORIZED_DB.  A ratio >= 0.75 (cutoff) tolerates
            roughly 1–3 character misreads per plate string, covering the most
            common Tesseract OCR confusions on number plates:
              '0' ↔ 'O',  '8' ↔ 'B',  '1' ↔ 'I',  '5' ↔ 'S',  '2' ↔ 'Z'
            The AUTHORIZED_DB key (not the raw OCR string) is returned so that
            downstream Gate 4 attribute lookup always uses the canonical plate.
        """
        self._bootstrap()
        binary = self._preprocess(frame)

        # Pre-compute the candidate list once per call (avoids repeated dict calls)
        valid_plates = list(AUTHORIZED_DB.keys())

        # Pass 1: PSM 8 — single-word mode, ideal for single-line car plates
        try:
            raw8   = pytesseract.image_to_string(binary, config=self._CFG_PSM8)
            plate8 = self._normalise(raw8)
            log.debug("[OCR] PSM8 raw=%r  norm=%r", raw8.strip(), plate8)
            if plate8:
                # Fuzzy lookup: returns a list of ≤1 best match above cutoff=0.75
                matches8 = difflib.get_close_matches(
                    plate8, valid_plates, n=1, cutoff=0.75
                )
                if matches8:
                    matched_plate = matches8[0]
                    if matched_plate == plate8:
                        log.info("[GATE3] OCR match (PSM8): %s", matched_plate)
                    else:
                        log.info(
                            "[GATE3] OCR match (Fuzzy PSM8: '%s' -> '%s')",
                            plate8, matched_plate,
                        )
                    return matched_plate
        except Exception as exc:
            log.warning("[OCR] PSM8 error: %s", exc)

        # Pass 2: PSM 6 — block mode, fallback for two-line two-wheeler plates
        try:
            raw6   = pytesseract.image_to_string(binary, config=self._CFG_PSM6)
            plate6 = self._normalise(raw6)
            log.debug("[OCR] PSM6 raw=%r  norm=%r", raw6.strip(), plate6)
            if plate6:
                # Same fuzzy threshold applied to the PSM 6 normalised output
                matches6 = difflib.get_close_matches(
                    plate6, valid_plates, n=1, cutoff=0.75
                )
                if matches6:
                    matched_plate = matches6[0]
                    if matched_plate == plate6:
                        log.info("[GATE3] OCR match (PSM6 fallback): %s",
                                 matched_plate)
                    else:
                        log.info(
                            "[GATE3] OCR match (Fuzzy PSM6: '%s' -> '%s')",
                            plate6, matched_plate,
                        )
                    return matched_plate
        except Exception as exc:
            log.warning("[OCR] PSM6 error: %s", exc)

        log.info("[GATE3] No authorised plate found.")
        return None


# =============================================================================
#  GATE 4 — ANTI-SPOOFING ENGINE  (YOLO + HSV Colour Analysis)
# =============================================================================
class AntiSpoofEngine(_ThreadSafeSingleton):
    """
    Singleton anti-spoofing engine combining YOLOv8n object detection and
    HSV colour analysis to defeat printed-plate cloning attacks.

    Threat model:
        An attacker affixes a printed photograph of a valid plate onto a
        different vehicle.  Gate 3 (OCR) alone grants access; Gate 4 cross-
        checks the physical vehicle's YOLO class and dominant body colour against
        the AUTHORIZED_DB record.  A conflict in EITHER attribute logs an
        ANTI_SPOOF_BLOCK flag and DENIES entry without any network transmission.

    YOLO verification:
        YOLOv8n (COCO pretrained) detects all objects.  The largest bounding box
        among VEHICLE_YOLO_CLASSES is selected as the primary vehicle.  Its COCO
        class label is compared against the DB "type" field.

    HSV colour verification:
        The central 40% crop of the detected bounding box (width × 40%, height ×
        40%, centred) is converted to HSV.  Seven canonical bins are tested via
        cv2.inRange(); the bin with the most matching pixels wins.  The central
        crop avoids the front grille (plate reflection skews hue readings) and
        the roof edge (shadowed at low camera angles).

    Both checks must pass for Gate 4 to return True.
    """

    def _bootstrap(self) -> None:
        """Lazily load YOLOv8n on first verification call."""
        if self._bootstrapped:
            return
        log.info("[GATE4] Loading YOLOv8n (first-run may auto-download ~6 MB)...")
        self._yolo = None
        try:
            # Deferred import: avoids PyTorch CUDA probe at module import time,
            # which can stall for 15-60 s on Windows with certain driver setups.
            from ultralytics import YOLO
            self._yolo = YOLO("yolov8n.pt")
            log.info("[GATE4] YOLOv8n loaded.")
        except ImportError:
            log.warning("[GATE4] ultralytics not installed — YOLO check disabled.")
        except Exception as exc:
            log.error("[GATE4] YOLOv8n load error: %s", exc)
        self._bootstrapped = True

    # ── HSV Dominant Colour Classifier ────────────────────────────────────
    @staticmethod
    def _dominant_colour(roi_bgr: np.ndarray) -> str | None:
        """
        Classify the dominant colour of a BGR crop using seven HSV colour bins.

        Process:
          1. Convert crop to HSV (separates hue from brightness — robust to
             lighting level changes).
          2. For each of the seven bins, build a binary mask via cv2.inRange()
             and count the matching pixels.
          3. Return the label of the bin with the highest pixel count.

        HSV is preferred over BGR because the Hue channel is invariant to
        illumination intensity changes (as long as the scene is not overexposed),
        making the classifier reliable across day/night transitions in a car park.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return None
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        best_colour, best_count = None, 0
        for name, ranges in HSV_COLOR_BINS.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for (lo, hi) in ranges:
                mask |= cv2.inRange(hsv, lo, hi)
            count = int(cv2.countNonZero(mask))
            if count > best_count:
                best_count  = count
                best_colour = name
        return best_colour

    # ── YOLO Vehicle Detector ──────────────────────────────────────────────
    def _detect_largest_vehicle(
        self, frame: np.ndarray
    ) -> tuple[str | None, tuple | None]:
        """
        Run YOLOv8n and return (class_label, (x1,y1,x2,y2)) for the largest
        vehicle bounding box, or (None, None) if none detected.
        """
        if self._yolo is None:
            return None, None
        results = self._yolo.predict(frame, conf=0.40, verbose=False)
        best_label, best_box, best_area = None, None, 0
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id not in VEHICLE_YOLO_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area  = area
                    best_label = VEHICLE_YOLO_CLASSES[cls_id]
                    best_box   = (x1, y1, x2, y2)
        return best_label, best_box

    # ── Public Verification API ────────────────────────────────────────────
    def verify(self, frame: np.ndarray, plate_key: str) -> bool:
        """
        Gate 4 entry point.

        Returns True only if BOTH the YOLO-detected vehicle type AND the HSV
        dominant body colour match the AUTHORIZED_DB record for plate_key.

        If YOLOv8n is unavailable, the engine FAILS CLOSED (returns False) to
        prevent unauthenticated access.  An ANTI_SPOOF_BLOCK is logged for any
        mismatch so that audit trails are complete.

        Args:
            frame     : Full-resolution BGR VGA frame from the ESP32-CAM.
            plate_key : AUTHORIZED_DB key returned by Gate 3.

        Returns:
            True  — type and colour match; access granted.
            False — mismatch; ANTI_SPOOF_BLOCK, access denied.
        """
        self._bootstrap()

        record          = AUTHORIZED_DB.get(plate_key, {})
        expected_type   = record.get("type",  "").lower()
        expected_colour = record.get("color", "").lower()

        # ── Step 1: YOLO vehicle-class check ──────────────────────────────
        detected_label, best_box = self._detect_largest_vehicle(frame)

        if detected_label is None:
            log.warning("[GATE4] ANTI_SPOOF_BLOCK — no vehicle detected by YOLO.")
            return False

        if detected_label != expected_type:
            log.warning(
                "[GATE4] ANTI_SPOOF_BLOCK — type mismatch: expected '%s', "
                "detected '%s'.", expected_type, detected_label
            )
            return False

        log.info("[GATE4] Vehicle type OK: %s", detected_label)

        # ── Step 2: HSV colour check on central 40% bounding-box crop ─────
        x1, y1, x2, y2 = best_box
        w, h = x2 - x1, y2 - y1
        cx   = (x1 + x2) // 2
        cy   = (y1 + y2) // 2

        # Central 40% crop: avoids grille/plate at the bottom and roof shadow
        crop_w = max(1, int(w * 0.40))
        crop_h = max(1, int(h * 0.40))
        cx1 = max(0, cx - crop_w // 2)
        cy1 = max(0, cy - crop_h // 2)
        cx2 = min(frame.shape[1], cx1 + crop_w)
        cy2 = min(frame.shape[0], cy1 + crop_h)
        roi = frame[cy1:cy2, cx1:cx2]

        detected_colour = self._dominant_colour(roi)
        log.info("[GATE4] HSV dominant colour: %s (expected: %s)",
                 detected_colour, expected_colour)

        if detected_colour != expected_colour:
            log.warning(
                "[GATE4] ANTI_SPOOF_BLOCK — colour mismatch: expected '%s', "
                "detected '%s'.", expected_colour, detected_colour
            )
            return False

        log.info("[GATE4] Anti-spoof check PASSED: %s %s.",
                 detected_colour, detected_label)
        return True


# =============================================================================
#  GDPR PRIVACY ANONYMISATION ENGINE
# =============================================================================
class GDPRAnonymiser(_ThreadSafeSingleton):
    """
    Singleton frame anonymiser compliant with GDPR Article 25 (Privacy by Design)
    and Recital 26 (rendering data subjects unidentifiable).

    Two categories of personal data are anonymised before any frame is saved:

    ── Faces → Gaussian blur (kernel 99×99, σ auto) ──────────────────────────
        Gaussian blur is a well-established pseudonymisation technique for face
        protection.  A 99×99 kernel completely destroys facial geometry at typical
        camera resolutions (640×480) without affecting vehicle body appearance.

    ── Licence plates → Mosaic pixelation (10×10 or 15×15 tile) ─────────────
        Pixelation performs block-average downscaling followed by nearest-
        neighbour upscaling.  Unlike Gaussian blur (a linear convolution that
        is theoretically invertible via Wiener deconvolution if the kernel
        parameters are known), mosaic pixelation is a NON-INVERTIBLE, LOSSY
        quantisation operation: the mean of each N×N tile irreversibly discards
        the individual pixel values within the tile.  No mathematical inverse
        exists because the many-to-one averaging collapses spatial frequencies
        below the tile Nyquist limit to zero.  This constitutes mathematically
        irreversible pseudonymisation compliant with GDPR Article 25
        (data minimisation) and Recital 26.

    Detection strategy:
        Faces  → YOLOv8n-face (primary, deep learning, high recall)
                 OpenCV Haar cascade (fallback, zero extra dependencies)
        Plates → OpenCV Russian-plate Haar cascade +
                 Aspect-ratio contour filter (catches plates the cascade misses)

    The authorised plate is ALSO pixelated in saved images: its text was
    already captured by OCR and persisted to parking_log.csv, so the raw
    pixel data is no longer required for auditing (GDPR Art. 5 — data minimisation).
    """

    def _bootstrap(self) -> None:
        if self._bootstrapped:
            return

        self._yolo_face      = None
        self._using_yolo_face = False

        # ── Face detector ────────────────────────────────────────────────
        try:
            from ultralytics import YOLO
            # yolov8n-face.pt downloads automatically from Ultralytics Hub.
            # For air-gapped deployments, pre-download and pass an absolute path.
            self._yolo_face       = YOLO("yolov8n-face.pt")
            self._using_yolo_face = True
            log.info("[GDPR] YOLOv8-face detector loaded.")
        except Exception as exc:
            log.info("[GDPR] YOLOv8-face unavailable (%s). Using Haar fallback.", exc)
            self._haar_face = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

        # ── Plate detector ────────────────────────────────────────────────
        self._haar_plate = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
        )
        if self._haar_plate.empty():
            log.warning("[GDPR] Plate Haar cascade not found — contour-only mode.")

        self._bootstrapped = True

    # ── Detectors ──────────────────────────────────────────────────────────
    def _detect_faces(self, frame: np.ndarray) -> list[tuple]:
        """Return [(x, y, w, h)] for all detected human faces."""
        h0, w0 = frame.shape[:2]
        scale  = min(1.0, 640.0 / w0)
        small  = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)))
        boxes  = []
        if self._using_yolo_face:
            results = self._yolo_face.predict(
                small, imgsz=640, conf=0.45, iou=0.45, verbose=False
            )
            for r in results:
                for b in r.boxes.xyxy.tolist():
                    x1, y1, x2, y2 = (int(v / scale) for v in b[:4])
                    boxes.append((x1, y1, x2 - x1, y2 - y1))
        else:
            grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            dets = self._haar_face.detectMultiScale(
                grey, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
            )
            for (fx, fy, fw, fh) in (dets if len(dets) > 0 else []):
                boxes.append((int(fx / scale), int(fy / scale),
                              int(fw / scale), int(fh / scale)))
        return boxes

    def _detect_plates(self, frame: np.ndarray) -> list[tuple]:
        """Return [(x, y, w, h)] for all visible licence plates."""
        grey  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = []

        # Stage 1: Haar cascade
        if not self._haar_plate.empty():
            dets = self._haar_plate.detectMultiScale(
                grey, scaleFactor=1.1, minNeighbors=4, minSize=(60, 20)
            )
            for d in (dets if len(dets) > 0 else []):
                boxes.append(tuple(d))

        # Stage 2: Aspect-ratio contour filter (supplements Haar for missed plates)
        blurred = cv2.GaussianBlur(grey, (5, 5), 0)
        edges   = cv2.Canny(blurred, 50, 200)
        cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        H, W    = frame.shape[:2]
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0:
                continue
            aspect = w / h
            area   = w * h
            # Plates have aspect ratio 2.0–6.5 and occupy > 0.3% of the frame
            if 2.0 <= aspect <= 6.5 and area > H * W * 0.003:
                boxes.append((x, y, w, h))

        return boxes

    # ── Anonymisation Primitives ───────────────────────────────────────────
    @staticmethod
    def _blur_region(frame: np.ndarray, x: int, y: int,
                     w: int, h: int, ksize: int = 99) -> None:
        """Apply strong Gaussian blur in-place to a rectangular ROI (faces)."""
        k = ksize if ksize % 2 == 1 else ksize + 1
        frame[y:y+h, x:x+w] = cv2.GaussianBlur(
            frame[y:y+h, x:x+w], (k, k), 0
        )

    @staticmethod
    def _pixelate_region(frame: np.ndarray, x: int, y: int,
                         w: int, h: int, tile: int = 10) -> None:
        """
        Apply mosaic pixelation in-place to a rectangular ROI (licence plates).

        The tile parameter selects the mosaic block size (10×10 or 15×15 px).

        GDPR Compliance — Mosaic Pixelation as Irreversible Pseudonymisation:
            Mosaic pixelation downscales the ROI to `tile`×`tile` using bilinear
            averaging (each output pixel = mean of tile²-pixel neighbourhood),
            then nearest-neighbour upscales it back to the original dimensions.
            This is a LOSSY, NON-INVERTIBLE quantisation: the spatial frequencies
            within each block are permanently erased.  Reversing this requires
            the original pixel values, which no longer exist in the anonymised
            image.  This satisfies the "rendering unidentifiable" criterion of
            GDPR Recital 26 and the data-minimisation obligation of Article 25.
        """
        if w < 1 or h < 1:
            return
        roi  = frame[y:y+h, x:x+w]
        tiny = cv2.resize(roi,  (tile, tile), interpolation=cv2.INTER_LINEAR)
        frame[y:y+h, x:x+w] = cv2.resize(
            tiny, (w, h), interpolation=cv2.INTER_NEAREST
        )

    # ── Public API ──────────────────────────────────────────────────────────
    def anonymise(self, frame: np.ndarray) -> np.ndarray:
        """
        Return a COPY of `frame` with all faces Gaussian-blurred and all
        licence plates mosaic-pixelated.  The input frame is never mutated.

        Step 1: Detect all faces  → apply 99×99 Gaussian blur.
        Step 2: Detect all plates → apply 10×10 mosaic pixelation.
        """
        self._bootstrap()
        out = frame.copy()

        face_boxes  = self._detect_faces(out)
        plate_boxes = self._detect_plates(out)

        for (x, y, w, h) in face_boxes:
            # Expand by 20 px to include hairline and chin
            x = max(0, x - 20);  y = max(0, y - 20)
            w = min(out.shape[1] - x, w + 40)
            h = min(out.shape[0] - y, h + 40)
            self._blur_region(out, x, y, w, h)

        for (x, y, w, h) in plate_boxes:
            self._pixelate_region(out, x, y, w, h, tile=10)

        log.info("[GDPR] Anonymised %d face(s) and %d plate(s).",
                 len(face_boxes), len(plate_boxes))
        return out


# =============================================================================
#  VLC / LI-FI FALLBACK BRANCH
# =============================================================================
class VLCReceiver(_ThreadSafeSingleton):
    """
    Singleton VLC / Li-Fi optical signal detector.

    Architecture — rolling brightness buffer + FFT analysis:

      Each incoming frame contributes one brightness sample b[n] = max(grey),
      stored with a monotonic timestamp in a deque of depth VLC_WINDOW_FRAMES
      (~3 s at 15 fps).  Thread safety: push_frame() (poller thread) and
      analyse() (VLC worker thread) both hold a Lock around deque access.

    Signal processing pipeline:

      1. b[n] extraction:
           b[n] = max pixel value of the Gaussian-smoothed greyscale frame.
           The maximum is used rather than the mean because a directional VLC
           transmitter (flashlight / LED) creates a localised bright spot; the
           mean is diluted by the dark background and would attenuate the signal.

      2. scipy.signal.detrend(vals, type="linear"):
           Removes the linear trend (DC offset + linear drift caused by ambient
           illumination changes such as cloud cover or headlight movement).
           Without detrending, a slowly brightening scene would produce a
           spurious low-frequency peak that could be misidentified as the VLC
           carrier.

      3. scipy.fft.rfft(vals_detrended):
           Single-sided real FFT.  rfft is used instead of the full fft because
           b[n] is a real-valued sequence; rfft returns N//2+1 complex bins,
           halving memory and arithmetic cost.  The resulting frequency axis is
           computed by rfftfreq(N, d=1/fps_est).

      4. Carrier detection:
           k_max  = argmax of |B[k]| for k >= 1  (skip DC bin k=0)
           f_peak = xf[k_max]                    (Hz)
           if |f_peak - VLC_TARGET_FREQ_HZ| <= VLC_FREQ_TOL_HZ
              AND |B[k_max]| > VLC_MIN_MAGNITUDE:
               → trigger /open-gate, completely bypassing OCR.
    """

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        if not hasattr(obj, "_deque"):
            obj._deque = collections.deque(maxlen=VLC_WINDOW_FRAMES)
            obj._lock  = threading.Lock()
        return obj

    def push_frame(self, frame: np.ndarray) -> None:
        """
        Extract b[n] (max greyscale brightness after 9×9 blur) and append
        (timestamp, b[n]) to the rolling buffer.

        Called from the HTTP poller thread on every received frame.
        """
        grey    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grey, (9, 9), 0)
        b_n     = float(np.max(blurred))   # b[n] per specification
        with self._lock:
            self._deque.append((time.monotonic(), b_n))

    def analyse(self) -> bool:
        """
        Apply detrend + rfft to the rolling buffer and return True if the
        VLC carrier frequency is detected with sufficient magnitude.

        Returns False if:
          - Fewer than 20 samples are buffered (< ~1.3 s of data).
          - Window duration < 2.0 s (frequency resolution too coarse).
          - Signal standard deviation < 5 counts (static / no illumination change).
        """
        with self._lock:
            if len(self._deque) < 20:
                return False
            times = np.array([t for t, _ in self._deque])
            vals  = np.array([v for _, v in self._deque])

        duration = times[-1] - times[0]
        if duration < 2.0:
            return False
        if np.std(vals) < 5.0:     # essentially static scene, no modulation
            return False

        fps_est = len(vals) / duration

        # Step 1: Detrend — remove ambient DC illumination shift
        vals_dt = scipy.signal.detrend(vals, type="linear")

        # Step 2: Real FFT (efficient for real-valued brightness signal)
        N   = len(vals_dt)
        yf  = rfft(vals_dt)               # complex spectrum B[k]
        xf  = rfftfreq(N, d=1.0 / fps_est)  # frequency axis in Hz
        mag = np.abs(yf)                  # |B[k]| magnitudes

        if len(mag) < 2:
            return False

        # Find dominant non-DC peak
        k_max    = int(np.argmax(mag[1:]) + 1)   # skip DC bin 0
        f_peak   = float(xf[k_max])               # dominant frequency (Hz)
        mag_peak = float(mag[k_max])              # |B[k_max]|

        log.debug("[VLC] f_peak=%.2f Hz  |B[k]|=%.1f", f_peak, mag_peak)

        # Carrier detection condition:
        #   |f_peak - 3.0 Hz| <= 0.8 Hz   AND   |B[k_max]| > 100
        if (abs(f_peak - VLC_TARGET_FREQ_HZ) <= VLC_FREQ_TOL_HZ
                and mag_peak > VLC_MIN_MAGNITUDE):
            log.info(
                "[VLC] Carrier detected: f=%.2f Hz  |B|=%.1f — gate bypass.",
                f_peak, mag_peak,
            )
            with self._lock:
                self._deque.clear()   # reset to prevent immediate re-trigger
            return True

        return False


# =============================================================================
#  EVENT LOGGER  (thread-safe CSV append)
# =============================================================================
_csv_lock = threading.Lock()

def _ensure_csv_header() -> None:
    """Create parking_log.csv with column headers if it does not exist."""
    if not os.path.isfile(LOG_CSV_PATH):
        with open(LOG_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["Timestamp", "PlateText", "AuthMethod", "Status"]
            )

def log_event(plate_text: str, auth_method: str, status: str) -> None:
    """
    Append one event row to parking_log.csv in a thread-safe manner.

    Columns:
      Timestamp  : ISO-8601 UTC timestamp (seconds precision).
      PlateText  : Normalised plate string, or "N/A" for VLC bypass events.
      AuthMethod : OCR_MATCH | VLC_FALLBACK | ANTI_SPOOF_BLOCK
      Status     : GRANTED | DENIED
    """
    _ensure_csv_header()
    row = [
        datetime.datetime.utcnow().isoformat(timespec="seconds"),
        plate_text,
        auth_method,
        status,
    ]
    with _csv_lock:
        with open(LOG_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
    log.info("[LOG] %s | %s | %s | %s", *row)


# =============================================================================
#  HTTP HELPERS
# =============================================================================
def _decode_jpeg(content: bytes) -> np.ndarray | None:
    """Decode a raw JPEG byte payload into an OpenCV BGR ndarray."""
    arr = np.frombuffer(content, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)   # None if decode fails

def _open_gate() -> None:
    """Fire-and-forget GET to /open-gate on the ESP32 barrier controller."""
    try:
        resp = requests.get(OPEN_GATE_URL, timeout=5)
        log.info("[GATE] /open-gate → HTTP %d", resp.status_code)
    except requests.RequestException as exc:
        log.error("[GATE] /open-gate request failed: %s", exc)

def _save_anonymised_frame(frame: np.ndarray, plate_text: str,
                            auth_method: str) -> None:
    """
    Anonymise `frame` via GDPRAnonymiser and write to ANON_SAVE_DIR.
    Filename: <YYYYMMDD_HHMMSS>_<plate>_<method>.jpg
    """
    anonymised = GDPRAnonymiser().anonymise(frame)
    ts    = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = f"{ts}_{plate_text}_{auth_method}.jpg"
    path  = os.path.join(ANON_SAVE_DIR, fname)
    cv2.imwrite(path, anonymised, [cv2.IMWRITE_JPEG_QUALITY, 85])
    log.info("[GDPR] Frame saved: %s", path)


# =============================================================================
#  FRAME PROCESSING BRANCHES  (Gate 3 + Gate 4)
# =============================================================================
def _process_ocr_and_antispoof(frame: np.ndarray) -> None:
    """
    Full Gate 3 + Gate 4 pipeline for one validated frame.

    Decision tree:
      Gate 3 OCR → plate found in AUTHORIZED_DB?
        YES → Gate 4 YOLO+HSV anti-spoof?
                PASS → /open-gate  + log(OCR_MATCH, GRANTED)
                FAIL → log(ANTI_SPOOF_BLOCK, DENIED)
        NO  → log(OCR_MATCH, DENIED)

    A GDPR-anonymised copy of the frame is saved in ALL branches
    to satisfy the data-controller logging obligation (GDPR Art. 5).
    """
    ocr_engine   = OCREngine()
    spoof_engine = AntiSpoofEngine()

    plate_key = ocr_engine.recognise(frame)

    if plate_key is None:
        log.info("[PIPELINE] Gate 3 miss — no authorised plate in frame.")
        log_event("UNKNOWN", "OCR_MATCH", "DENIED")
        _save_anonymised_frame(frame, "UNKNOWN", "OCR_DENIED")
        return

    log.info("[PIPELINE] Gate 3 PASS: %s — running Gate 4 anti-spoof...", plate_key)
    antispoof_ok = spoof_engine.verify(frame, plate_key)

    if antispoof_ok:
        log.info("[PIPELINE] Gate 4 PASS — granting entry: %s", plate_key)
        _open_gate()
        log_event(plate_key, "OCR_MATCH", "GRANTED")
        _save_anonymised_frame(frame, plate_key, "OCR_MATCH")
    else:
        log.warning("[PIPELINE] Gate 4 FAIL — ANTI_SPOOF_BLOCK: %s", plate_key)
        log_event(plate_key, "ANTI_SPOOF_BLOCK", "DENIED")
        _save_anonymised_frame(frame, plate_key, "ANTI_SPOOF_BLOCK")


# =============================================================================
#  VLC WORKER THREAD
# =============================================================================
def _vlc_worker(stop_event: threading.Event) -> None:
    """
    Background daemon thread that polls VLCReceiver.analyse() every 200 ms.

    When a VLC/Li-Fi carrier is confirmed it fires /open-gate immediately,
    completely bypassing Gate 3 (OCR) and Gate 4 (anti-spoof).  The VLC
    channel provides an out-of-band authentication path for deliveries or
    emergency access where the vehicle plate may not be legible.

    The shared VLCReceiver brightness buffer is populated concurrently by
    VLCReceiver.push_frame() inside the HTTP polling loop — no additional
    inter-thread communication is needed beyond the deque and its lock.

    stop_event: threading.Event set by the main thread to signal shutdown.
    """
    vlc = VLCReceiver()
    log.info("[VLC] Worker thread started (polling VLCReceiver every 200 ms).")
    while not stop_event.is_set():
        try:
            if vlc.analyse():
                log.info("[VLC] Li-Fi bypass — triggering gate without OCR.")
                _open_gate()
                log_event("N/A", "VLC_FALLBACK", "GRANTED")
                # Inhibit re-triggering: sleep for the full gate-hold window + margin
                time.sleep(6.0)
        except Exception as exc:
            log.error("[VLC] Worker exception: %s", exc)
        time.sleep(0.2)   # 5 analysis attempts per second
    log.info("[VLC] Worker thread stopped.")


# =============================================================================
#  MAIN HTTP POLLING LOOP
# =============================================================================
def run_polling_loop(stop_event: threading.Event | None = None) -> None:
    """
    Core HTTP polling engine.

    Every POLL_INTERVAL_S (2.0 s) a GET request is sent to /smart-capture.

    Response handling:
      HTTP 204 No Content :
          The ESP32's Gate 1 + Gate 2 pipeline found no validated vehicle.
          Action: pass.  No JPEG is transmitted, minimising Wi-Fi bandwidth
          and preventing thermal throttling of the ESP32-CAM module.

      HTTP 200 OK :
          The ESP32 has a JPEG from a Gate-1-AND-Gate-2-validated vehicle.
          Action:
            1. Decode the JPEG payload into a BGR ndarray.
            2. Push the frame into the VLCReceiver buffer (Li-Fi branch A).
            3. Spawn a daemon thread for Gate 3 + Gate 4 (Branch B).
               The daemon thread means the 2-second poll cadence is never
               blocked by Tesseract or YOLO inference time.

      Other status codes: logged as unexpected; loop continues.

    Args:
        stop_event : threading.Event.  Set to True for graceful shutdown.
                     Pass None to run indefinitely until KeyboardInterrupt.
    """
    session = requests.Session()   # Reuse TCP connection to reduce per-poll overhead
    log.info("[POLLER] HTTP polling started → %s  (%.1f s interval)",
             SMART_CAPTURE_URL, POLL_INTERVAL_S)

    vlc_receiver = VLCReceiver()

    while True:
        if stop_event is not None and stop_event.is_set():
            log.info("[POLLER] Stop signal received — exiting.")
            break

        loop_start = time.monotonic()

        try:
            response = session.get(SMART_CAPTURE_URL, timeout=5.0)

            if response.status_code == 204:
                # No validated vehicle — pass silently to conserve bandwidth
                log.debug("[POLLER] 204 — pipeline inactive, no vehicle at gate.")

            elif response.status_code == 200:
                log.info("[POLLER] 200 — %d byte JPEG payload received.",
                         len(response.content))

                frame = _decode_jpeg(response.content)
                if frame is None:
                    log.error("[POLLER] JPEG decode failed — discarding frame.")
                else:
                    # ── Normal pipeline ────────────────────
                    # Branch A: VLC brightness buffer update (non-blocking O(1))
                    vlc_receiver.push_frame(frame)

                    # Branch B: Gate 3 + Gate 4 in a background daemon thread.
                    # Copy the frame so the original stays in the VLCReceiver
                    # deque without the processing thread mutating it.
                    t = threading.Thread(
                        target=_process_ocr_and_antispoof,
                        args=(frame.copy(),),
                        daemon=True,
                        name="OCR-AntiSpoof",
                    )
                    t.start()

            else:
                log.warning("[POLLER] Unexpected HTTP %d.", response.status_code)

        except requests.exceptions.ConnectionError:
            log.warning("[POLLER] ESP32 unreachable — retrying in %.1f s.",
                        POLL_INTERVAL_S)
        except requests.exceptions.Timeout:
            log.warning("[POLLER] Request timed out (5 s).")
        except Exception as exc:
            log.error("[POLLER] Unhandled error: %s", exc)

        # Honour the 2-second cadence even if processing consumed some time
        elapsed = time.monotonic() - loop_start
        time.sleep(max(0.0, POLL_INTERVAL_S - elapsed))


# =============================================================================
#  ENTRY POINT
# =============================================================================
def main() -> None:
    """
    Application bootstrap:

      1. Create parking_log.csv header row if the file does not exist.
      2. Launch the VLC/Li-Fi worker daemon thread.
      3. Enter the HTTP polling loop (blocks until KeyboardInterrupt or
         stop_event is set).
      4. On exit, signal the VLC thread and wait up to 2 s for clean shutdown.
    """
    _ensure_csv_header()

    stop_event = threading.Event()

    vlc_thread = threading.Thread(
        target=_vlc_worker,
        args=(stop_event,),
        daemon=True,
        name="VLC-Worker",
    )
    vlc_thread.start()
    log.info("[MAIN] VLC/Li-Fi worker thread started.")
    log.info("[MAIN] Polling ESP32-CAM at %s", SMART_CAPTURE_URL)
    log.info("[MAIN] Gate unlock: /open-gate → %s", OPEN_GATE_URL)
    log.info("[MAIN] Anonymised frames → ./%s/", ANON_SAVE_DIR)
    log.info("[MAIN] Audit log → ./%s", LOG_CSV_PATH)

    try:
        run_polling_loop(stop_event=stop_event)
    except KeyboardInterrupt:
        log.info("[MAIN] Keyboard interrupt — shutting down.")
    finally:
        stop_event.set()
        vlc_thread.join(timeout=2.0)
        log.info("[MAIN] Shutdown complete.")


# =============================================================================
#  BACKEND BRIDGE — called by backend/main.py AI worker thread
# =============================================================================
def open_gate(sio=None) -> None:
    """
    Public wrapper for _open_gate() so backend/main.py can trigger the servo
    directly (e.g. from the force-open REST endpoint).
    """
    _open_gate()


def run_ai_pipeline(sio=None) -> None:
    """
    Entry-point called by the AI worker thread in backend/main.py.

    Runs the existing HTTP polling loop and keeps shared_frame_jpg /
    dashboard_state in sync so the FastAPI layer can serve the video feed
    and telemetry via Socket.IO.
    """
    global shared_frame_jpg, dashboard_state

    stop_event = threading.Event()

    # Start VLC/Li-Fi daemon exactly as main() does.
    vlc_thread = threading.Thread(
        target=_vlc_worker,
        args=(stop_event,),
        daemon=True,
        name="VLC-Worker",
    )
    vlc_thread.start()
    log.info("[AI-PIPELINE] VLC/Li-Fi worker thread started.")
    log.info("[AI-PIPELINE] Polling ESP32-CAM at %s", SMART_CAPTURE_URL)

    session = requests.Session()

    while True:
        loop_start = time.monotonic()
        try:
            response = session.get(SMART_CAPTURE_URL, timeout=5.0)

            if response.status_code == 200:
                log.info("[AI-PIPELINE] 200 — %d byte JPEG received.",
                         len(response.content))

                frame = _decode_jpeg(response.content)
                if frame is None:
                    log.error("[AI-PIPELINE] JPEG decode failed — discarding.")
                else:
                    # Update shared MJPEG frame for the video feed endpoint.
                    _, buf = cv2.imencode(".jpg", frame)
                    with _frame_lock:
                        shared_frame_jpg = buf.tobytes()

                    VLCReceiver().push_frame(frame)
                    t = threading.Thread(
                        target=_pipeline_with_state_update,
                        args=(frame.copy(),),
                        daemon=True,
                        name="OCR-AntiSpoof",
                    )
                    t.start()

            elif response.status_code == 204:
                log.debug("[AI-PIPELINE] 204 — no vehicle at gate.")

            else:
                log.warning("[AI-PIPELINE] Unexpected HTTP %d.",
                            response.status_code)

        except requests.exceptions.ConnectionError:
            log.warning("[AI-PIPELINE] ESP32 unreachable — retrying in %.1f s.",
                        POLL_INTERVAL_S)
            with _frame_lock:
                shared_frame_jpg = b""  # clear stale frame
        except requests.exceptions.Timeout:
            log.warning("[AI-PIPELINE] Request timed out (5 s).")
        except Exception as exc:
            log.error("[AI-PIPELINE] Unhandled error: %s", exc)

        elapsed = time.monotonic() - loop_start
        time.sleep(max(0.0, POLL_INTERVAL_S - elapsed))


def _pipeline_with_state_update(frame: np.ndarray) -> None:
    """
    Runs Gate 3 + Gate 4, then mirrors the result into dashboard_state
    and emits the appropriate Socket.IO events.
    """
    global dashboard_state

    ocr_engine   = OCREngine()
    spoof_engine = AntiSpoofEngine()

    plate_key = ocr_engine.recognise(frame)

    if plate_key is None:
        log.info("[PIPELINE] Gate 3 miss — no authorised plate in frame.")
        log_event("UNKNOWN", "OCR_MATCH", "DENIED")
        _save_anonymised_frame(frame, "UNKNOWN", "OCR_DENIED")
        dashboard_state.update({
            "gate_status": "DENIED",
            "plate_text":  "",
            "authorized":  False,
        })
        _emit(None, "gate_event", {
            "status":    "DENIED",
            "plate":     "",
            "timestamp": time.strftime("%H:%M:%S"),
        })
        return

    antispoof_ok = spoof_engine.verify(frame, plate_key)

    if antispoof_ok:
        log.info("[PIPELINE] Gate 4 PASS — granting entry: %s", plate_key)
        _open_gate()
        log_event(plate_key, "OCR_MATCH", "GRANTED")
        _save_anonymised_frame(frame, plate_key, "OCR_MATCH")
        dashboard_state.update({
            "gate_status": "OPEN",
            "plate_text":  plate_key,
            "authorized":  True,
        })
        _emit(None, "gate_event", {
            "status":    "OPEN",
            "plate":     plate_key,
            "timestamp": time.strftime("%H:%M:%S"),
        })
    else:
        log.warning("[PIPELINE] Gate 4 FAIL — ANTI_SPOOF_BLOCK: %s", plate_key)
        log_event(plate_key, "ANTI_SPOOF_BLOCK", "DENIED")
        _save_anonymised_frame(frame, plate_key, "ANTI_SPOOF_BLOCK")
        dashboard_state.update({
            "gate_status": "DENIED",
            "plate_text":  plate_key,
            "authorized":  False,
        })
        _emit(None, "gate_event", {
            "status":    "DENIED",
            "plate":     plate_key,
            "timestamp": time.strftime("%H:%M:%S"),
        })


if __name__ == "__main__":
    main()
