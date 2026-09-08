# Elderly-Fall-Detection
# 🚨 IoT-Based Elderly Fall Detection System

An intelligent **IoT and AI/ML-based fall detection system** designed to monitor elderly individuals and detect potential fall events in real time.

The system uses an **ESP32**, **MPU6050 accelerometer and gyroscope**, GPS, and Wi-Fi to collect motion and location data. Fall detection is performed using both **firmware-based threshold detection** and a **2D CNN-based machine learning model** for improved activity and fall classification.

## ✨ Features

* 🚨 Real-time fall detection
* 📊 Motion sensing using the MPU6050
* 🧠 **2D CNN-based fall detection and classification**
* ⚡ Firmware-based acceleration and gyroscope threshold detection
* 📍 GPS-based location tracking
* 📡 Wi-Fi communication with a backend server
* 📤 Real-time sensor data transmission using HTTP and JSON
* ⏱️ Alert cooldown mechanism to reduce repeated notifications

## 🔌 Hardware Connections

### MPU6050 → ESP32

| MPU6050 Pin | ESP32 Pin |
| ----------- | --------- |
| VCC         | 3.3V      |
| GND         | GND       |
| SDA         | GPIO 21   |
| SCL         | GPIO 22   |

### GPS Module → ESP32

| GPS Pin | ESP32 Pin    |
| ------- | ------------ |
| TX      | GPIO 16      |
| RX      | GPIO 17      |
| VCC     | Power Supply |
| GND     | GND          |

## 🛠️ Technologies Used

**Hardware:** ESP32, MPU6050, GPS Module
**Embedded:** Arduino, C/C++
**Communication:** Wi-Fi, HTTP, JSON
**AI/ML:** Python, 2D CNN
**Backend:** REST API

## 🚀 Working

# 🚨 IoT-Based Elderly Fall Detection System

An intelligent **IoT and AI/ML-based fall detection system** designed to monitor elderly individuals and detect potential fall events in real time.

The system uses an **ESP32**, **MPU6050 accelerometer and gyroscope**, GPS, and Wi-Fi to collect motion and location data. Fall detection is performed using both **firmware-based threshold detection** and a **2D CNN-based machine learning model** for improved activity and fall classification.

## ✨ Features

* 🚨 Real-time fall detection
* 📊 Motion sensing using the MPU6050
* 🧠 **2D CNN-based fall detection and classification**
* ⚡ Firmware-based acceleration and gyroscope threshold detection
* 📍 GPS-based location tracking
* 📡 Wi-Fi communication with a backend server
* 📤 Real-time sensor data transmission using HTTP and JSON
* ⏱️ Alert cooldown mechanism to reduce repeated notifications

## 🔌 Hardware Connections

### MPU6050 → ESP32

| MPU6050 Pin | ESP32 Pin |
| ----------- | --------- |
| VCC         | 3.3V      |
| GND         | GND       |
| SDA         | GPIO 21   |
| SCL         | GPIO 22   |

### GPS Module → ESP32

| GPS Pin | ESP32 Pin    |
| ------- | ------------ |
| TX      | GPIO 16      |
| RX      | GPIO 17      |
| VCC     | Power Supply |
| GND     | GND          |

> **Note:** The exact GPS module power connection depends on the module being used.

## 🛠️ Technologies Used

**Hardware:** ESP32, MPU6050, GPS Module
**Embedded:** Arduino, C/C++
**Communication:** Wi-Fi, HTTP, JSON
**AI/ML:** Python, 2D CNN
**Backend:** REST API

## 🚀 Working

The system operates as a continuous sensing, processing, and communication pipeline. After startup, the ESP32 initializes the I2C interface used by the MPU6050 and opens a serial connection with the GPS module. It then connects to the configured Wi-Fi network and prepares the backend API endpoint for transmitting sensor readings.

The MPU6050 continuously measures acceleration along the X, Y, and Z axes, as well as angular velocity around those axes. These readings describe both the movement and orientation of the user. The ESP32 periodically reads the sensor values and calculates the overall acceleration magnitude:

```text
Acceleration magnitude = √(Ax² + Ay² + Az²)
```

This value helps identify sudden changes in motion. During a possible fall, the system may observe a rapid acceleration spike caused by the initial movement, followed by a sudden reduction in acceleration when the person impacts the ground. A significant change in gyroscope values can also indicate a rapid rotation or change in body orientation.

The ESP32 first performs local threshold-based detection. If the acceleration magnitude, gyroscope magnitude, or related motion values exceed predefined limits, the event is marked as a possible fall. This local decision provides a fast first-level response without waiting for the backend or machine-learning model. To avoid sending repeated alerts for the same incident, the firmware uses an alert cooldown period. Once an alert is triggered, additional alerts are temporarily suppressed until the cooldown expires.

At regular intervals, the ESP32 packages the sensor readings into a JSON object. The payload may include acceleration values, gyroscope values, acceleration magnitude, gyroscope magnitude, timestamp information, and the current GPS coordinates. The ESP32 sends this data to the backend server through an HTTP request over Wi-Fi. The backend can store the readings, display them on a monitoring dashboard, and forward them to the machine-learning pipeline.

The GPS module continuously outputs NMEA sentences through the serial interface. The ESP32 parses these messages and extracts the latitude, longitude, and, when available, satellite and location-fix information. If a valid GPS fix is available during a suspected fall, the coordinates are attached to the alert payload. This allows a caregiver or monitoring service to identify the approximate location of the elderly person. If GPS data is unavailable indoors or during startup, the system can still transmit the fall event with the latest valid coordinates or an indication that no fix is currently available.

For intelligent classification, the collected motion data is arranged into time-based samples or windows. Instead of analyzing only one sensor reading, the 2D CNN receives a sequence containing multiple acceleration and gyroscope measurements. This allows the model to learn motion patterns such as walking, sitting, standing, lying down, sudden movement, and falling. The sensor window is normalized and formatted according to the input shape used during model training. The 2D CNN then extracts spatial and temporal features through convolutional layers and produces a classification result.

The machine-learning output can be combined with the firmware decision. For example, a threshold event may be treated as a confirmed fall only when the CNN also predicts a fall with sufficient confidence. Alternatively, the firmware can immediately generate a preliminary alert while the CNN performs secondary verification. This combination improves response time while reducing false alarms caused by activities such as jumping, quickly sitting down, or dropping the device.

When a fall is confirmed, the backend can record the event and trigger an emergency notification. The notification may contain the detected activity, confidence score, timestamp, sensor values, and GPS location. A caregiver can then review the event and take appropriate action. The system continues monitoring after the alert cooldown period and resumes normal data transmission.

Overall, the workflow is:

1. Initialize the ESP32, MPU6050, GPS, and Wi-Fi connection.
2. Read acceleration and gyroscope values continuously.
3. Calculate motion magnitudes and monitor sudden changes.
4. Apply firmware-based threshold detection.
5. Parse GPS data and obtain the latest valid location.
6. Group sensor readings into time-based windows.
7. Send sensor data to the backend using HTTP and JSON.
8. Analyze motion windows using the 2D CNN model.
9. Combine threshold and CNN results to classify the activity.
10. Generate and transmit an alert when a fall is confirmed.
11. Apply a cooldown period to prevent duplicate alerts.
12. Continue monitoring for future events.

## 👩‍💻 Author

**Trushali Phatak**
Aspiring Space Systems Engineer | RF & Satellite Communications | Embedded Systems | AI/ML
