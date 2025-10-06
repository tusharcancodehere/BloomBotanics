"""
🌱 BloomBotanics Sensor Modules
Handles all sensor inputs for agricultural monitoring
"""

__version__ = "2.0.0"
__author__ = "BloomBotanics Team"

from .dht22_sensor import DHT22Sensor
from .soil_moisture import SoilMoistureSensor
from .rain_sensor import RainSensor

try:
    from .ai_camera import AICamera
except ImportError:
    print("⚠️ AI Camera module not available - AI detection disabled")
    AICamera = None

__all__ = ['DHT22Sensor', 'SoilMoistureSensor', 'RainSensor', 'AICamera']
