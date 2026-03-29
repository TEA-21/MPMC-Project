// ============================================================
// ESP32-CAM Smart LPR System — Esp32.cpp
// Feature 1: TinyML Edge-Side Image Optimization (Edge Impulse)
// Features: Ultrasonic Pre-filter → TinyML Classifier → OCR Server
// ============================================================

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include <ESP32Servo.h>

// ┌─────────────────────────────────────────────────────────┐
// │  STEP 1: Add this include AFTER installing the Arduino  │
// │  library exported from Edge Impulse (.zip file).        │
// │  The exact name matches your Edge Impulse project name. │
// └─────────────────────────────────────────────────────────┘
#include <ESP32_CAM_Vehicle_Classifier_inferencing.h>
#include "edge-impulse-sdk/dsp/image/image.hpp"

// ----------------------------------------------------------
// WiFi & Hardware Config
// ----------------------------------------------------------
const char* ssid     = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

#define SERVO_PIN 14
#define TRIG_PIN  13
#define ECHO_PIN  12

// Distance threshold: vehicle must be within 1.5 metres
#define DETECTION_THRESHOLD_CM 150

// ┌─────────────────────────────────────────────────────────┐
// │  STEP 2: Set your confidence threshold.                 │
// │  0.70 means the model must be ≥70% sure it is a        │
// │  vehicle before allowing the image through.             │
// │  Raise it to reduce false positives;                   │
// │  lower it if real vehicles are being suppressed.        │
// └─────────────────────────────────────────────────────────┘
#define EI_VEHICLE_CONFIDENCE_THRESHOLD 0.70f

// ----------------------------------------------------------
// AI Thinker ESP32-CAM Pin Map
// ----------------------------------------------------------
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

WebServer server(80);
Servo     gateServo;

// ----------------------------------------------------------
// Camera Initialisation
// ----------------------------------------------------------
void setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Start with higher quality; we downscale only for inference
  if (psramFound()) {
    config.frame_size   = FRAMESIZE_VGA;   // 640×480 for OCR quality
    config.jpeg_quality = 10;
    config.fb_count     = 2;
  } else {
    config.frame_size   = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count     = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[CAM] Init failed: 0x%x\n", err);
  }
}

// ----------------------------------------------------------
// Gate 1: Ultrasonic Pre-filter (Hardware)
// ----------------------------------------------------------
bool isVehiclePresent() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000); // 30ms timeout
  if (duration == 0) return false;

  float distance = duration * 0.034f / 2.0f;
  Serial.printf("[ULTRASONIC] Distance: %.1f cm\n", distance);
  return (distance > 0 && distance < DETECTION_THRESHOLD_CM);
}

// ----------------------------------------------------------
// Gate 2: TinyML Image Classifier (Software)
// ┌─────────────────────────────────────────────────────────┐
// │  HOW IT WORKS:                                          │
// │  1. Capture a small JPEG frame (96×96) for inference.   │
// │  2. Decompress JPEG → raw RGB pixels.                   │
// │  3. Edge Impulse SDK resizes/grayscales internally.     │
// │  4. run_classifier() executes MobileNetV1 inference.    │
// │  5. Returns true only if "vehicle" score ≥ threshold.  │
// └─────────────────────────────────────────────────────────┘
// ----------------------------------------------------------

// ┌─────────────────────────────────────────────────────────┐
// │  IMPORTANT: This static pixel buffer holds the          │
// │  decompressed RGB888 frame for the 96×96 inference.     │
// │  EI_CLASSIFIER_INPUT_WIDTH / HEIGHT come from the       │
// │  generated Edge Impulse header — they equal 96.         │
// └─────────────────────────────────────────────────────────┘
static uint8_t ei_pixel_buf[EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT * 3];

// Callback that the Edge Impulse SDK calls to get pixel data
// "offset" is the flat pixel index, "length" is how many pixels to fill
static int ei_camera_get_data(size_t offset, size_t length, float *out_ptr) {
  size_t pixel_ix   = offset * 3;   // 3 bytes per RGB pixel
  size_t pixels_left = length;
  size_t out_ptr_ix  = 0;

  while (pixels_left != 0) {
    // Pack RGB bytes into a single uint32: 0x00RRGGBB
    out_ptr[out_ptr_ix] = (ei_pixel_buf[pixel_ix]     << 16) |  // R
                          (ei_pixel_buf[pixel_ix + 1] <<  8) |  // G
                           ei_pixel_buf[pixel_ix + 2];           // B
    out_ptr_ix++;
    pixel_ix   += 3;
    pixels_left--;
  }
  return 0; // 0 = success in Edge Impulse SDK convention
}

bool runTinyMLVehicleCheck() {
  // --- 2a. Capture a small frame just for inference ---
  // Temporarily switch to 96×96 to keep inference RAM minimal
  sensor_t* s = esp_camera_sensor_get();
  s->set_framesize(s, FRAMESIZE_96X96);
  delay(100); // Let the sensor stabilise at the new resolution

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[TinyML] Camera grab failed");
    s->set_framesize(s, FRAMESIZE_VGA); // Restore
    return false; // Safety: if we can't get a frame, skip inference
  }

  // --- 2b. Decompress JPEG → raw RGB888 pixels ---
  //  fmt2rgb888 decodes the JPEG into our static pixel buffer.
  bool converted = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, ei_pixel_buf);
  esp_camera_fb_return(fb); // Return frame buffer immediately to free memory

  // Restore full resolution for the final OCR-quality capture
  s->set_framesize(s, FRAMESIZE_VGA);

  if (!converted) {
    Serial.println("[TinyML] JPEG decode failed — passthrough");
    return true; // Fail open: if decode fails, let OCR server try
  }

  // --- 2c. Build the Edge Impulse signal struct ---
  // This tells the SDK HOW to access your pixel data (via callback above)
  ei::signal_t signal;
  signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
  signal.get_data     = &ei_camera_get_data;

  // --- 2d. Run the MobileNetV1 classifier ---
  ei_impulse_result_t result = { 0 };
  EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false /* debug */);

  if (err != EI_IMPULSE_OK) {
    Serial.printf("[TinyML] run_classifier error: %d — passthrough\n", err);
    return true; // Fail open: let the server decide
  }

  // --- 2e. Parse results and log them ---
  Serial.println("[TinyML] Classification results:");
  float vehicle_score = 0.0f;

  for (uint8_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
    Serial.printf("  %-12s: %.4f\n",
                  result.classification[ix].label,
                  result.classification[ix].value);

    // Match the label your Edge Impulse project uses for vehicles
    if (strcmp(result.classification[ix].label, "vehicle") == 0) {
      vehicle_score = result.classification[ix].value;
    }
  }

  // --- 2f. Apply the confidence threshold gate ---
  if (vehicle_score >= EI_VEHICLE_CONFIDENCE_THRESHOLD) {
    Serial.printf("[TinyML] ✅ VEHICLE confirmed (score=%.2f) — forwarding image\n",
                  vehicle_score);
    return true;
  } else {
    Serial.printf("[TinyML] ❌ No vehicle (score=%.2f) — suppressed\n",
                  vehicle_score);
    return false;
  }
}

// ----------------------------------------------------------
// Main HTTP Handler: /smart-capture
// Dual-gate architecture: Ultrasonic → TinyML → OCR Server
// ----------------------------------------------------------
void handleSmartCapture() {

  // ═══════════════════════════════════════════
  // GATE 1: Ultrasonic Hardware Pre-filter
  // Fast & zero-image-cost. Filters ~70% of idle polls.
  // ═══════════════════════════════════════════
  if (!isVehiclePresent()) {
    Serial.println("[HANDLER] Gate 1 FAIL — ultrasonic: no object nearby");
    server.send(204, "text/plain", "No object in detection zone.");
    return;
  }
  Serial.println("[HANDLER] Gate 1 PASS — object detected by ultrasonic");

  // ═══════════════════════════════════════════
  // GATE 2: TinyML Image Classifier (Feature 1)
  // Visual confirmation that the object is a vehicle.
  // Runs at ~96×96 for speed; filters bikes, pedestrians, etc.
  // ═══════════════════════════════════════════
  if (!runTinyMLVehicleCheck()) {
    Serial.println("[HANDLER] Gate 2 FAIL — TinyML: object is NOT a vehicle");
    server.send(204, "text/plain", "No vehicle confirmed by TinyML.");
    return;
  }
  Serial.println("[HANDLER] Gate 2 PASS — TinyML: vehicle confirmed");

  // ═══════════════════════════════════════════
  // CAPTURE: Full-resolution image for OCR
  // Now that both gates have passed, capture the
  // highest-quality JPEG for the Python server's
  // license plate OCR + YOLO attribute check.
  // ═══════════════════════════════════════════
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[HANDLER] Full-res capture failed");
    server.send(500, "text/plain", "Camera capture failed");
    return;
  }

  Serial.printf("[HANDLER] Sending %u bytes to server\n", fb->len);
  server.send_P(200, "image/jpeg", (const char*)fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

// ----------------------------------------------------------
// Gate Handler: /open-gate
// ----------------------------------------------------------
void handleOpenGate() {
  gateServo.write(90);
  delay(3000);
  gateServo.write(0);
  server.send(200, "text/plain", "Gate Opened");
}

// ----------------------------------------------------------
// Arduino Setup
// ----------------------------------------------------------
void setup() {
  Serial.begin(115200);
  Serial.println("\n[BOOT] ESP32-CAM Smart LPR starting...");

  // Print Edge Impulse model info for verification
  Serial.printf("[EI]   Model: %s\n",    EI_CLASSIFIER_PROJECT_NAME);
  Serial.printf("[EI]   Input: %dx%d\n", EI_CLASSIFIER_INPUT_WIDTH,
                                          EI_CLASSIFIER_INPUT_HEIGHT);
  Serial.printf("[EI]   Labels: %d\n",   EI_CLASSIFIER_LABEL_COUNT);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  gateServo.setPeriodHertz(50);
  gateServo.attach(SERVO_PIN, 500, 2400);
  gateServo.write(0); // Closed position

  WiFi.begin(ssid, password);
  Serial.print("[WIFI] Connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.printf("\n[WIFI] Connected. IP: %s\n", WiFi.localIP().toString().c_str());

  setupCamera();

  server.on("/smart-capture", handleSmartCapture);
  server.on("/open-gate",     handleOpenGate);
  server.begin();
  Serial.println("[HTTP] Server started on port 80");
}

// ----------------------------------------------------------
// Arduino Loop
// ----------------------------------------------------------
void loop() {
  server.handleClient();
}