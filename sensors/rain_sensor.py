"""Rain Detection Sensor Module"""

import RPi.GPIO as GPIO
import time
from utils.logger import get_logger

class RainSensor:
    def __init__(self, digital_pin=27, analog_pin=22):
        self.logger = get_logger(__name__)
        self.digital_pin = digital_pin
        self.analog_pin = analog_pin
        self.setup_gpio()
    
    def setup_gpio(self):
        """Setup GPIO pins for rain sensor"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.digital_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            self.logger.info(f"Rain sensor initialized - Digital: GPIO{self.digital_pin}")
        except Exception as e:
            self.logger.error(f"Rain sensor setup error: {e}")
    
    def read_rain_status(self):
        """Read rain detection status"""
        try:
            # Digital rain detection (LOW = rain detected)
            digital_value = not GPIO.input(self.digital_pin)  # Inverted logic
            
            # Determine intensity based on digital reading
            if digital_value:
                intensity = "Moderate"  # Can be enhanced with analog reading
            else:
                intensity = "None"
            
            return {
                'rain_detected': digital_value,
                'intensity': intensity,
                'digital_value': digital_value,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Rain sensor reading error: {e}")
            return {
                'rain_detected': False,
                'intensity': 'Unknown',
                'digital_value': False,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'error'
            }
    
    def get_rain_history(self, duration_minutes=60):
        """Get rain detection history (placeholder for future enhancement)"""
        # This would read from stored data to determine recent rain patterns
        return {
            'rain_in_last_hour': False,
            'total_rain_time': 0,
            'last_rain_time': None
        }
    
    def cleanup(self):
        try:
            GPIO.cleanup([self.digital_pin])
        except:
            pass
