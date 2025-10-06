# BloomBotanics Simple API

Base URL: http://your_pi_ip:8080/api

## Main Endpoints

### System Status
GET /api/status

Response:
{
  "status": "running",
  "temperature": 25.3,
  "humidity": 65.2,
  "soil_moisture": 45.7,
  "rain_detected": false,
  "irrigation_active": false
}

### Sensor Data
GET /api/sensors

Response:
{
  "timestamp": "2025-09-28T07:00:00Z",
  "temperature": 25.3,
  "humidity": 65.2, 
  "soil_moisture": 45.7,
  "rain_detected": false
}

### Irrigation Control
POST /api/irrigation/start
{
  "duration": 30
}

POST /api/irrigation/stop

Response:
{
  "status": "success",
  "message": "Irrigation started",
  "duration": 30,
  "started_at": "2025-09-28T07:00:00Z"
}

### Camera
POST /api/camera/capture

GET /api/camera/latest

Response:
{
  "status": "success",
  "filename": "crop_20250928_070000.jpg",
  "captured_at": "2025-09-28T07:00:00Z",
  "url": "/images/crop_20250928_070000.jpg"
}

### AI Detection
GET /api/detection/recent

Response:
{
  "detections": [
    {
      "timestamp": "2025-09-28T06:45:00Z",
      "object_type": "person",
      "confidence": 0.85,
      "threat_level": "critical",
      "image_path": "detections/threat_20250928_064500.jpg"
    }
  ]
}

### SMS Status
GET /api/sms/status

Response:
{
  "gsm_connected": true,
  "signal_strength": 85,
  "sms_sent_today": 8,
  "last_sms_sent": "2025-09-28T06:45:00Z"
}

### System Health
GET /api/health

Response:
{
  "overall_status": "healthy",
  "cpu_temperature": 65.2,
  "memory_usage_percent": 45.6,
  "disk_usage_percent": 23.1,
  "network_connected": true
}

### Configuration
GET /api/config

Response:
{
  "farm_name": "BloomBotanics Farm",
  "phone_number": "+91987***3210",
  "auto_irrigation": true,
  "ai_detection_enabled": true,
  "thresholds": {
    "temperature_min": 15.0,
    "temperature_max": 35.0,
    "soil_moisture_min": 30.0
  }
}

## Python Examples

### Basic Sensor Monitoring
import requests
import time

# Get sensor data
response = requests.get('http://192.168.1.100:8080/api/sensors')
data = response.json()

print(f"Temperature: {data['temperature']}°C")
print(f"Humidity: {data['humidity']}%")
print(f"Soil Moisture: {data['soil_moisture']}%")

# Start irrigation if soil is dry
if data['soil_moisture'] < 30:
    requests.post('http://192.168.1.100:8080/api/irrigation/start', 
                  json={'duration': 30})
    print("Irrigation started!")

### Continuous Monitoring
import requests
import time

API_BASE = "http://192.168.1.100:8080/api"

while True:
    try:
        # Get system status
        status = requests.get(f"{API_BASE}/status").json()
        
        print(f"Temp: {status['temperature']}°C")
        print(f"Humidity: {status['humidity']}%")
        print(f"Soil: {status['soil_moisture']}%")
        print(f"Rain: {'Yes' if status['rain_detected'] else 'No'}")
        print(f"Irrigation: {'Active' if status['irrigation_active'] else 'Off'}")
        print("-" * 30)
        
        time.sleep(60)  # Check every minute
        
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(10)

## JavaScript Examples

### Simple Dashboard
// Update dashboard every 30 seconds
async function updateDashboard() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        document.getElementById('temperature').textContent = data.temperature + '°C';
        document.getElementById('humidity').textContent = data.humidity + '%';
        document.getElementById('soil').textContent = data.soil_moisture + '%';
        document.getElementById('rain').textContent = data.rain_detected ? 'Yes' : 'No';
        
    } catch (error) {
        console.error('Update failed:', error);
    }
}

setInterval(updateDashboard, 30000);

### Manual Irrigation Control
// Start irrigation button
async function startIrrigation() {
    try {
        const response = await fetch('/api/irrigation/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ duration: 30 })
        });
        
        const result = await response.json();
        alert(`Irrigation ${result.status}: ${result.message}`);
        
    } catch (error) {
        alert('Irrigation failed: ' + error.message);
    }
}

## cURL Examples

### Get System Status
curl http://192.168.1.100:8080/api/status

### Start Manual Irrigation
curl -X POST http://192.168.1.100:8080/api/irrigation/start \
     -H "Content-Type: application/json" \
     -d '{"duration": 30}'

### Capture Photo
curl -X POST http://192.168.1.100:8080/api/camera/capture

## Error Responses

### Standard Error Format
{
  "error": {
    "code": "SENSOR_UNAVAILABLE",
    "message": "DHT22 sensor not responding",
    "timestamp": "2025-09-28T07:00:00Z"
  }
}

### HTTP Status Codes
- 200 - Success
- 400 - Bad Request
- 404 - Not Found  
- 500 - Internal Server Error
- 503 - Service Unavailable

### Common Error Codes
- SENSOR_UNAVAILABLE - Sensor not responding
- IRRIGATION_ACTIVE - Cannot start - already running
- GSM_DISCONNECTED - SMS functionality unavailable
- CAMERA_ERROR - Camera capture failed
- INVALID_DURATION - Invalid irrigation duration

## Rate Limits
- General API: 60 requests per minute
- Control Actions: 10 requests per minute
- Image Capture: 5 requests per minute

Note: This is a simplified API for the BloomBotanics system. The main interface is through SMS alerts and Pi Connect remote access, with this API providing programmatic control for custom integrations.
