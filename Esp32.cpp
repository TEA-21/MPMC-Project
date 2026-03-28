#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include <ESP32Servo.h>

// ===========================
// Wi-Fi Credentials
// ===========================
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// ===========================
// Pin Definitions
// ===========================
// Note: The ESP32-CAM has limited free GPIOs. 
// Assuming the following are used based on standard availability:
#define SERVO_PIN 14 
#define TRIG_PIN 13
#define ECHO_PIN 12

// ===========================
// Camera Model (AI Thinker)
// ===========================
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
Servo gateServo;

// ===========================
// Setup Camera
// ===========================
void setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG; // MJPEG streaming
  
  // Frame parameters
  if(psramFound()){
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  // Initialize camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }
}

// ===========================
// HTTP Endpoints
// ===========================

// Endpoint to capture a single JPEG frame (used by Python edge server)
void handleCapture() {
  camera_fb_t * fb = NULL;
  fb = esp_camera_fb_get();
  if (!fb) {
    server.send(500, "text/plain", "Camera capture failed");
    return;
  }
  server.send_P(200, "image/jpeg", (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

// Endpoint to trigger the servo motor (Gate Mechanism)
void handleOpenGate() {
  Serial.println("Signal received. Opening gate...");
  gateServo.write(90); // Rotate to 90 degrees (open)
  delay(3000);         // Keep open for 3 seconds
  gateServo.write(0);  // Return to 0 degrees (closed)
  server.send(200, "text/plain", "Gate Opened and Closed");
}

// Endpoint to read Ultrasonic Distance (Obstacle Detection)
void handleDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  
  long duration = pulseIn(ECHO_PIN, HIGH);
  float distance = duration * 0.034 / 2; // Convert to cm
  
  String html = String(distance);
  server.send(200, "text/plain", html);
}

// ===========================
// Main Setup & Loop
// ===========================
void setup() {
  Serial.begin(115200);
  
  // Setup Sensor Pins
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  
  // Setup Servo
  gateServo.setPeriodHertz(50); 
  gateServo.attach(SERVO_PIN, 500, 2400); 
  gateServo.write(0); // Ensure gate starts closed

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");
  Serial.print("ESP32-CAM IP Address: ");
  Serial.println(WiFi.localIP());

  setupCamera();

  // Define routing
  server.on("/capture", handleCapture); // Python server requests this URL
  server.on("/open-gate", handleOpenGate); // Python server triggers this on success
  server.on("/distance", handleDistance); // Optional check for vehicle presence
  
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  server.handleClient();
}