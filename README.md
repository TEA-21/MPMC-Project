# MPMC-Project: Edge-AI License Plate Recognition System

<div align="center">

⚡ **Cyberpunk Smart Parking Management System** ⚡

**Real-time Vehicle Detection • GDPR-Compliant Privacy • Multi-Factor Authentication • Li-Fi Unlock**

[Features](#-features) • [Architecture](#-architecture) • [Setup](#-setup) • [Usage](#-usage) • [Tech Stack](#-tech-stack)

</div>

---

## 🎯 Overview

MPMC-Project is an **edge-AI powered License Plate Recognition (LPR) system** designed for smart parking management. It combines **computer vision (YOLOv8), optical character recognition (OCR)**, and **innovative Li-Fi (visible light communication)** to provide secure, privacy-first vehicle access control.

### Key Innovation: Multi-Layer Security
- **OCR-based plate recognition** with character normalization
- **Visual authentication** (color + vehicle type verification)
- **Li-Fi flashlight unlock** (3 Hz frequency detection via FFT)
- **GDPR-compliant privacy** (automatic face & plate anonymization)
- **Real-time predictive analytics** (occupancy forecasting)

---

## ✨ Features

### 1. **License Plate Recognition (LPR)**
- Tesseract OCR with image preprocessing
- Automatic character normalization (O→0, I→1, L→1)
- Confidence-based validation
- Real-time plate extraction from video feed

### 2. **Multi-Factor Authentication**
- **Primary**: Database lookup of authorized plates
- **Secondary**: YOLOv8 vehicle type detection (Sedan, SUV, Truck, etc.)
- **Tertiary**: HSV-based color matching (Red, White, Blue, Black, Silver, etc.)
- Prevents printed-plate cloning attacks

### 3. **Privacy & GDPR Compliance**
- **YOLOv8-face model** for face detection → Gaussian blur anonymization
- **Two-stage plate detection** (Haar cascade + contour filtering) → Mosaic pixelation
- Fallback to Haar cascades if ultralytics unavailable
- All personally identifiable data removed from stored images

### 4. **Li-Fi (Visible Light Communication) Unlock**
- **Flashlight pattern recognition** via FFT analysis
- Detects 3 Hz stroboscopic signals (±0.8 Hz tolerance)
- Fallback unlock: sustained brightness >200 intensity for 2s
- Pulse counting with 3-pulse detection or simple light override
- Perfect for hands-free access in emergency situations

### 5. **Predictive Analytics & Occupancy Forecasting**
- Polynomial regression (degree 4) on historical parking logs
- 24-hour occupancy prediction with rush-hour detection
- Matplotlib-based cyberpunk dashboard visualization
- Synthetic data generation for cold-start scenarios

### 6. **Real-Time WebSocket Dashboard**
- Live MJPEG video feed (25 fps cap)
- Socket.IO telemetry updates (100 ms polling)
- Li-Fi intensity visualization with pulse counter
- Gate status display (IDLE / OPEN / DENIED)
- Force-open & system-reset controls
- Dark-themed cyberpunk UI

### 7. **Hardware Integration**
- **ESP32 Sensor Node**: Ultrasonic presence detection + servo gate control
- Distance threshold: 1.5m (150cm)
- HTTP-based communication (check-presence / open-gate endpoints)
- Configurable WiFi connectivity

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     ESP32 IoT Node                          │
│  • Ultrasonic Presence Sensor (HC-SR04)                     │
│  • Servo Motor Gate Controller (90° swing)                  │
│  • WiFi HTTP Server (check-presence, open-gate)            │
└────────────┬────────────────────────────────────────────────┘
             │ HTTP REST
             ▼
┌─────────────────────────────────────────────────────────────┐
│           Backend: FastAPI + Socket.IO (Async)             │
│  • Port 8080 (API docs: /docs)                             │
│  • ASGI factory pattern (Python 3.13+ compatible)          │
│  • Telemetry loop (100 ms state updates)                   │
│  • CORS enabled for cross-origin requests                  │
└────────────┬────────────────────────────────────────────────┘
             │ WebSocket + REST
             ▼
┌─────────────────────────────────────────────────────────────┐
│    Frontend: Vue 3 + Vite (Dev: Port 5173, Prod: 8080)    │
│  • Real-time video stream consumption                       │
│  • Live Socket.IO telemetry display                        │
│  • Control panel (force-open, system-reset)                │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────┐         ┌──────────────────────┐
│   AI Pipeline (OS   │         │  Data Persistence   │
│    Background      │         │    (Async CSV       │
│     Thread)        │         │     Logging)        │
│                    │         │                     │
│ • Video Capture    │         │ • parking_log.csv   │
│ • YOLOv8 Vehicle   │         │ • OCR Results       │
│ • Tesseract OCR    │         │ • Entry Timestamps  │
│ • Privacy Filter   │         │                     │
│ • Li-Fi Detector   │         │                     │
│ • Auth Check       │         │                     │
│ • Gate Control     │         │                     │
└──────────────────────┘         └──────────────────────┘
```

### Data Flow

1. **ESP32** detects vehicle → sends `DETECTED` signal
2. **AI Pipeline** captures frame from webcam
3. **OCR Module** extracts license plate text
4. **Database Lookup** checks authorization
5. **Visual Auth** verifies vehicle color & type
6. **Privacy Filter** anonymizes faces & plates
7. **Gate Control** opens servo if authorized
8. **Logging** records entry with timestamp
9. **Dashboard** streams live updates via WebSocket

---

## 🚀 Setup

### Prerequisites

**Hardware:**
- PC/Raspberry Pi with USB webcam
- ESP32 microcontroller with ultrasonic sensor & servo
- WiFi connectivity

**Software:**
- Python 3.10+ (tested on 3.10 - 3.13)
- Node.js 16+ (for frontend development)
- pip & npm package managers

### Installation

#### 1. Clone Repository
```bash
git clone https://github.com/TEA-21/MPMC-Project.git
cd MPMC-Project
```

#### 2. Backend Setup
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Key packages:
# - fastapi (web framework)
# - uvicorn (ASGI server)
# - python-socketio (WebSocket)
# - opencv-python (cv2)
# - pytesseract (OCR)
# - ultralytics (YOLOv8)
# - scikit-learn (ML models)
# - pandas (data analysis)
```

**Tesseract OCR Setup** (required for plate recognition):
- **Windows**: Download installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS**: `brew install tesseract`
- **Linux**: `apt-get install tesseract-ocr`

After installation, update path in `server.py` if needed:
```python
pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
```

#### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev  # Development server on http://localhost:5173
```

#### 4. ESP32 Configuration
Upload `Esp32.cpp` to your ESP32 board using Arduino IDE:
```cpp
const char *ssid = "Your_WiFi_SSID";
const char *password = "Your_WiFi_Password";
#define SERVO_PIN 14
#define TRIG_PIN 13
#define ECHO_PIN 12
#define DETECTION_THRESHOLD_CM 150
```

Then update `server.py` with ESP32 IP:
```python
ESP32_IP = "10.212.43.45"  # Your ESP32's IP address
```

---

## 💻 Usage

### Quick Start

**Option 1: Unified Development Launch** (Recommended)
```bash
python dev.py
# Starts both backend (port 8080) and frontend (port 5173)
# Press Ctrl+C to shut down gracefully
```

**Option 2: Manual Launch**

Backend:
```bash
python -m uvicorn backend.main:create_app --factory --host 0.0.0.0 --port 8080
# Or: python dashboard.py
```

Frontend:
```bash
cd frontend && npm run dev
```

**Option 3: Dashboard Mode** (Legacy - Flask)
```bash
python dashboard.py
# Opens http://localhost:5000
# Runs AI pipeline in background thread
```

### Running Analytics

```bash
python predictive_analytics.py
# Generates 24-hour occupancy forecast
# Displays matplotlib dashboard
```

### Demo Mode

Test without ESP32 or webcam:
```python
# In server.py, set:
TEST_MODE = True
# Uses test_car.jpg for all frames
```

### Logging

Authorized vehicle entries are logged to `parking_log.csv`:
```csv
timestamp,entry
2026-04-15 10:32:15.123456,1
2026-04-15 14:47:22.654321,1
```

---

## 📊 API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check (`{"status": "ok"}`) |
| `GET` | `/state` | Dashboard state snapshot |
| `GET` | `/video_feed` | MJPEG video stream |
| `POST` | `/force-open` | Manually open gate (override) |
| `POST` | `/system-reset` | Reset dashboard state |

### WebSocket Events

| Event | Direction | Payload |
|-------|-----------|---------|
| `connect` | ← Server | Initial state snapshot |
| `telemetry_update` | ← Server | `{intensity, pulse_count, lifi_state, plate_text, gate_status, authorized}` |
| `ocr_result` | ← Server | `{plate_text, authorized}` |
| `lifi_update` | ← Server | `{intensity, state, pulse_count}` |
| `gate_event` | ← Server | `{status, plate, timestamp}` |
| `log_line` | ← Server | `{message, level}` |
| `state_snapshot` | ← Server | Full dashboard state |
| `system_reset` | ← Server | Signals reset to connected clients |

### Database (Authorized Vehicles)

Located in `server.py`:
```python
AUTHORIZED_DB = {
    "TN01AB1234": ("White", "SUV"),
    "KA05XY9876": ("Red", "Sedan")
}
```

Add more entries to authorize additional vehicles.

---

## ⚙️ Configuration

### Core Parameters

**OCR Processing** (`server.py`, line 839):
```python
# Upscale factor for Tesseract accuracy
gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Otsu thresholding for binary text
_, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
```

**Li-Fi Detection** (`server.py`, line 287-288):
```python
target_freq = 3.0        # Target 3 Hz flashlight frequency
freq_tolerance = 0.8     # Allow ±0.8 Hz variation
LIFI_THRESHOLD = 200     # Brightness intensity threshold
```

**Privacy Filter** (`server.py`, line 156):
```python
ksize = 99               # Gaussian blur kernel (faces)
mosaic = 10              # Pixelation grid size (plates)
```

**Camera** (`server.py`, line 671):
```python
cap = _open_camera_with_timeout(index=1, use_dshow=True, timeout_sec=10)
# index=0 fallback if index=1 fails
```

**ESP32 Detection** (`server.py`, line 22):
```python
DETECTION_THRESHOLD_CM = 150  # Vehicle must be within 1.5m
```

---

## 🏃 Performance Notes

### Hardware Requirements
- **GPU**: Recommended for real-time YOLOv8 inference (RTX 2080 Ti capable of ~30 fps)
- **CPU**: i7/Ryzen 7 minimum for CPU fallback
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 150MB+ for models (yolov8n.pt ~6.5MB, yolov8n-face.pt ~6MB)

### Optimization Strategies
1. **YOLOv8 Model**: Using `yolov8n` (nano) for speed vs. accuracy tradeoff
2. **Frame Preprocessing**: 2x upscaling for OCR, then downscaled if >1000px width
3. **MJPEG Compression**: 25 fps cap + 80% JPEG quality
4. **Threading**: Deferred PyTorch import from module scope to avoid startup hang
5. **Li-Fi FFT**: Rolling 60-frame window (3s @ ~20 fps) for frequency analysis

### Known Limitations
- **Windows CUDA**: Can hang 15-60s on first PyTorch import due to DirectShow enumeration
- **Tesseract**: Accuracy ~85-95% depending on image quality and lighting
- **Li-Fi**: Requires manual 3 Hz flashing or sustained bright light source
- **Network**: ESP32 HTTP calls may timeout in poor WiFi conditions

---

## 🔒 Security & Privacy

### GDPR Compliance
✅ **Data Minimization**: Anonymized images stored; only plate text retained  
✅ **Face Protection**: All human faces blurred (Gaussian 99×99 kernel)  
✅ **Plate Anonymization**: Mosaic pixelation (10×10 grid) for secondary plates  
✅ **Primary Plate**: Even authorized plates anonymized in stored images  

### Authentication Layers
1. **Plate Database**: Whitelist-based authorization
2. **Visual Verification**: YOLOv8 + HSV color matching
3. **Li-Fi Override**: Optional emergency access via flashlight signal

### API Security
- ⚠️ **Current**: No authentication on REST/WebSocket endpoints (edge device assumed secure network)
- 🔒 **Recommended**: Add JWT tokens or API key authentication for production deployment

---

## 📁 Project Structure

```
MPMC-Project/
├── server.py                      # Core AI pipeline (OCR, detection, auth)
├── dashboard.py                   # Flask-SocketIO dashboard (legacy)
├── demo_mode.py                   # Historical demo/testing utilities
├── dev.py                         # Unified dev launcher (uvicorn + Vite)
├── predictive_analytics.py        # 24-hour occupancy forecasting
├── grab_dat.py                    # Dataset preparation utility
│
├── Esp32.cpp                      # ESP32 firmware (ultrasonic + servo)
├── parking_log.csv                # CSV log of vehicle entries
│
├── backend/
│   ├── main.py                    # FastAPI factory + Socket.IO server
│   └── __init__.py
│
├── frontend/                      # Vue 3 + Vite SPA
│   ├── src/
│   ├── public/
│   ├── package.json
│   ├── vite.config.js
│   └── index.html
│
├── static/                        # Flask static assets (legacy)
├── templates/                     # Flask HTML templates (legacy)
├── vehicle/                       # Vehicle classification data
│
├── yolov8n.pt                     # YOLOv8 nano model (~6.5 MB)
├── test_car.jpg                   # Test frame for demo mode
├── processed_ocr.jpg              # Last OCR preprocessing output
└── README.md                      # This file
```

---

## 🛠️ Troubleshooting

### Common Issues

**1. "PyTorch/CUDA hangs on import"**
- **Cause**: DirectShow device enumeration on Windows
- **Fix**: Update to latest driver; use `_open_camera_with_timeout()` (already implemented)

**2. "Tesseract not found"**
- **Cause**: Missing installation or PATH issue
- **Fix**: Reinstall and set `pytesseract.pytesseract_cmd` path

**3. "YOLOv8 model download fails"**
- **Cause**: No internet access or Hub token expired
- **Fix**: Download `.pt` files manually, place in project root, update path in code

**4. "ESP32 connection timeout"**
- **Cause**: Wrong IP, WiFi disconnected, or firewall
- **Fix**: Verify IP with `Serial Monitor`, check WiFi, add firewall rule

**5. "OCR accuracy too low"**
- **Cause**: Poor lighting, tilted plates, or dirty camera lens
- **Fix**: Improve lighting, adjust upscaling factor (fx/fy), clean lens

**6. "WebSocket connection refused"**
- **Cause**: CORS misconfiguration or port in use
- **Fix**: Check `dev.py` port bindings, verify allowed origins in `main.py`

---

## 📦 Dependencies

### Python Packages
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-socketio==5.10.0
python-engineio==4.8.0
flask==3.0.0
flask-socketio==5.3.5
flask-cors==4.0.0
opencv-python==4.8.1.78
pytesseract==0.3.10
ultralytics==8.0.195
torch==2.1.1
torchvision==0.16.1
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.26.2
matplotlib==3.8.2
scipy==1.11.4
requests==2.31.0
```

### System Dependencies
- **Windows**: Tesseract-OCR, Visual C++ Redistributable
- **macOS**: Tesseract (via Homebrew)
- **Linux**: tesseract-ocr, libopencv-dev

### Node.js Packages
```json
{
  "dependencies": {
    "vue": "^3.3.0",
    "vite": "^5.0.0"
  }
}
```
