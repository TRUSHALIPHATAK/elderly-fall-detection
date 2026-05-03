#include <Wire.h>
#include <MPU6050.h>
#include <TinyGPSPlus.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <math.h>

const char* WIFI_SSID     = "Trush";
const char* WIFI_PASSWORD = "lemon123";
const char* SERVER_URL    = "http://10.160.124.178:5000/api/sensor-data";
const char* DEVICE_ID     = "ESP32-001";

// ── Firmware fall detection thresholds ──
#define ACCEL_FALL_THRESHOLD  25.0f   // m/s²
#define GYRO_FALL_THRESHOLD   300.0f  // deg/s
#define COOLDOWN_MS           10000   // 10 seconds between firmware alerts

MPU6050 mpu;
TinyGPSPlus gps;
HardwareSerial gpsSerial(2);

unsigned long last_send_ms   = 0;
unsigned long last_fall_ms   = 0;  // cooldown tracker

void wifiConnect() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 40) {
    delay(500); Serial.print("."); tries++;
  }
  if (WiFi.status() == WL_CONNECTED)
    Serial.println("\nWiFi OK → " + WiFi.localIP().toString());
  else
    Serial.println("\nWiFi FAILED");
}

void rawToPhysical(int16_t rax, int16_t ray, int16_t raz,
                   int16_t rgx, int16_t rgy, int16_t rgz,
                   float &ax, float &ay, float &az,
                   float &gx, float &gy, float &gz) {
  ax = (rax / 2048.0f) * 9.81f;
  ay = (ray / 2048.0f) * 9.81f;
  az = (raz / 2048.0f) * 9.81f;
  gx = rgx / 16.4f;
  gy = rgy / 16.4f;
  gz = rgz / 16.4f;
}

void setup() {
  Serial.begin(115200);
  delay(500);

  Wire.begin(21, 22);
  mpu.initialize();
  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_16);
  mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_2000);
  if (!mpu.testConnection()) {
    Serial.println("MPU6050 NOT FOUND");
    while (1) delay(1000);
  }
  Serial.println("MPU6050 OK");

  gpsSerial.begin(9600, SERIAL_8N1, 16, 17);
  Serial.println("GPS started");

  wifiConnect();
  Serial.println("Ready — sending data every 200ms\n");
}

void loop() {
  unsigned long now = millis();

  while (gpsSerial.available())
    gps.encode(gpsSerial.read());

  if (now - last_send_ms >= 200) {
    last_send_ms = now;

    int16_t rax, ray, raz, rgx, rgy, rgz;
    mpu.getMotion6(&rax, &ray, &raz, &rgx, &rgy, &rgz);

    float ax, ay, az, gx, gy, gz;
    rawToPhysical(rax, ray, raz, rgx, rgy, rgz,
                  ax, ay, az, gx, gy, gz);

    float accel_mag = sqrtf(ax*ax + ay*ay + az*az);
    float gyro_mag  = sqrtf(gx*gx + gy*gy + gz*gz);

    // ── Firmware fall detection with cooldown ──
    bool fw_fall = false;
    if (accel_mag > ACCEL_FALL_THRESHOLD && gyro_mag > GYRO_FALL_THRESHOLD) {
      if (now - last_fall_ms > COOLDOWN_MS) {
        fw_fall = true;
        last_fall_ms = now;
        Serial.println("⚠️  FIRMWARE FALL DETECTED");
      }
    }

    bool   gps_fixed = true;
    double lat = 19.046191668643647;
    double lng = 72.87107215236664;

    Serial.println("──────────────────────────────");
    Serial.printf("Accel  ax=%.2f  ay=%.2f  az=%.2f  mag=%.2f\n", ax, ay, az, accel_mag);
    Serial.printf("Gyro   gx=%.1f  gy=%.1f  gz=%.1f  mag=%.1f\n", gx, gy, gz, gyro_mag);
    Serial.printf("GPS FIX lat=%.5f lng=%.5f\n", lat, lng);
    Serial.printf("fw_fall=%s\n", fw_fall ? "YES" : "no");

    if (WiFi.status() != WL_CONNECTED) { wifiConnect(); return; }

    HTTPClient http;
    http.begin(SERVER_URL);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(3000);

    StaticJsonDocument<384> doc;
    doc["device_id"] = DEVICE_ID;
    doc["ax"] = ax;  doc["ay"] = ay;  doc["az"] = az;
    doc["gx"] = gx;  doc["gy"] = gy;  doc["gz"] = gz;
    doc["accel_mag"] = accel_mag;
    doc["gyro_mag"]  = gyro_mag;
    doc["fw_fall"]   = fw_fall;   // ← firmware fall flag
    doc["lat"]       = lat;
    doc["lng"]       = lng;
    doc["gps_fixed"] = gps_fixed;
    doc["battery"]   = 90;

    String body;
    serializeJson(doc, body);

    int code = http.POST(body);
    Serial.printf("HTTP → %d\n\n", code);
    http.end();
  }
}