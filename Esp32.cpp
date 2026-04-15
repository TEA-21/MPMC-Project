// ============================================================
// ESP32 Smart LPR Node (Standard ESP32 Edition)
// Features: Ultrasonic Presence Detection & Servo Control
// ============================================================

#include <ESP32Servo.h>
#include <WebServer.h>
#include <WiFi.h>

// ----------------------------------------------------------
// WiFi & Hardware Config
// ----------------------------------------------------------
const char *ssid = "Redmi Note 14 5G";
const char *password = "Taanush14";

// Using standard GPIO pins
#define SERVO_PIN 14
#define TRIG_PIN 13
#define ECHO_PIN 12

// Distance threshold: vehicle must be within 1.5 metres
#define DETECTION_THRESHOLD_CM 150

WebServer server(80);
Servo gateServo;

// ----------------------------------------------------------
// Gate 1: Ultrasonic Pre-filter
// ----------------------------------------------------------
bool isVehiclePresent() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  if (duration == 0)
    return false;

  float distance = duration * 0.034f / 2.0f;
  return (distance > 0 && distance < DETECTION_THRESHOLD_CM);
}

// ----------------------------------------------------------
// Main HTTP Handlers
// ----------------------------------------------------------
void handleCheckPresence() {
  // Instead of sending an image, we now just send a boolean state
  if (isVehiclePresent()) {
    Serial.println("[HANDLER] Object detected in zone.");
    server.send(200, "text/plain", "DETECTED");
  } else {
    server.send(204, "text/plain", "NONE");
  }
}

void handleOpenGate() {
  Serial.println("[HANDLER] Opening Gate...");
  gateServo.write(90);
  delay(3000);
  gateServo.write(0);
  server.send(200, "text/plain", "Gate Opened");
}

// ----------------------------------------------------------
// Arduino Setup & Loop
// ----------------------------------------------------------
void setup() {
  Serial.begin(115200);
  Serial.println("\n[BOOT] ESP32 Sensor Node starting...");

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  gateServo.setPeriodHertz(50);
  gateServo.attach(SERVO_PIN, 500, 2400);
  gateServo.write(0);

  WiFi.begin((char *)ssid, (char *)password);
  Serial.print("[WIFI] Connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.printf("\n[WIFI] Connected. IP: %s\n",
                WiFi.localIP().toString().c_str());

  // Updated endpoint name to reflect its new purpose
  server.on("/check-presence", handleCheckPresence);
  server.on("/open-gate", handleOpenGate);
  
  server.begin();
  Serial.println("[HTTP] Server started on port 80");
}

void loop() { 
  server.handleClient(); 
}