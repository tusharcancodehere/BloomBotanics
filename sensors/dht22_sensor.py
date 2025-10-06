"""DHT22 Temperature and Humidity Sensor Module"""

import time
import board
import adafruit_dht
from utils.logger import get_logger

class DHT22Sensor:
    def __init__(self, pin=4):
        self.logger = get_logger(__name__)
        self.pin = pin
        try:
            self.sensor = adafruit_dht.DHT22(getattr(board, f'D{pin}'))
            self.logger.info(f"DHT22 sensor initialized on GPIO {pin}")
        except Exception as e:
            self.logger.error(f"DHT22 initialization failed: {e}")
            self.sensor = None
    
    def read_data(self):
        """Read temperature and humidity with retry logic"""
        if not self.sensor:
            return None
        
        for attempt in range(3):
            try:
                temperature = self.sensor.temperature
                humidity = self.sensor.humidity
                
                if temperature is not None and humidity is not None:
                    return {
                        'temperature': round(temperature, 1),
                        'humidity': round(humidity, 1),
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'status': 'success'
                    }
            except RuntimeError as e:
                self.logger.warning(f"DHT22 read attempt {attempt + 1}: {e}")
                time.sleep(2)
        
        self.logger.error("DHT22 reading failed after all attempts")
        return {
            'temperature': 0.0,
            'humidity': 0.0,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'error'
        }
    
    def cleanup(self):
        if self.sensor:
            self.sensor.exit()
