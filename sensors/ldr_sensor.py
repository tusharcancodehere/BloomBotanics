#!/usr/bin/env python3
"""
LDR Light Sensor Module
BloomBotanics - October 2025
"""

import RPi.GPIO as GPIO
import time
from utils.logger import get_logger


class LDRSensor:
    """LDR (Light Dependent Resistor) for ambient light detection"""
    
    def __init__(self, pin=24):
        """
        Initialize LDR sensor
        Args:
            pin: GPIO pin number (default: 24)
        """
        self.logger = get_logger(__name__)
        self.pin = pin
        
        self.setup_sensor()
    
    def setup_sensor(self):
        """Setup GPIO for LDR sensor"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.IN)
            self.logger.info(f"LDR sensor initialized on GPIO {self.pin}")
        except Exception as e:
            self.logger.error(f"LDR sensor setup error: {e}")
    
    def read_data(self):
        """Read light level from LDR sensor"""
        try:
            # Digital reading (HIGH = BRIGHT, LOW = DARK)
            digital_value = GPIO.input(self.pin)
            light_status = 'BRIGHT' if digital_value == GPIO.HIGH else 'DARK'
            
            # Estimate light percentage (approximation for digital sensor)
            light_percent = 80.0 if digital_value == GPIO.HIGH else 20.0
            
            return {
                'light_level': light_status,
                'light_percent': light_percent,
                'raw_value': digital_value,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"LDR sensor reading error: {e}")
            return {
                'light_level': 'UNKNOWN',
                'light_percent': 0.0,
                'raw_value': 0,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'error'
            }
    
    def is_daylight(self):
        """Check if it's daylight"""
        data = self.read_data()
        return data['light_level'] == 'BRIGHT'
    
    def cleanup(self):
        try:
            GPIO.cleanup([self.pin])
        except:
            pass
