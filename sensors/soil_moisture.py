#!/usr/bin/env python3
"""
Dual Soil Moisture Sensor Module with ADC Support
BloomBotanics - Updated October 2025
"""

import RPi.GPIO as GPIO
import time
import spidev
from utils.logger import get_logger


class SoilMoistureSensor:
    def __init__(self, pin1=22, pin2=23, use_adc=False, adc_channel1=0, adc_channel2=1):
        """
        Initialize dual soil moisture sensors
        Args:
            pin1: GPIO pin for Soil Sensor 1 (default: 22)
            pin2: GPIO pin for Soil Sensor 2 (default: 23)
            use_adc: Use MCP3008 ADC for analog reading (default: False)
            adc_channel1: ADC channel for Sensor 1 (default: 0)
            adc_channel2: ADC channel for Sensor 2 (default: 1)
        """
        self.logger = get_logger(__name__)
        self.pin1 = pin1
        self.pin2 = pin2
        self.use_adc = use_adc
        self.adc_channel1 = adc_channel1
        self.adc_channel2 = adc_channel2
        self.spi = None
        
        self.setup_sensor()
    
    def setup_sensor(self):
        """Setup GPIO or SPI for dual soil moisture sensors"""
        try:
            if self.use_adc:
                # Setup SPI for MCP3008 ADC
                self.spi = spidev.SpiDev()
                self.spi.open(0, 0)  # SPI port 0, device 0
                self.spi.max_speed_hz = 1350000
                self.logger.info("Soil sensors using MCP3008 ADC (2 channels)")
            else:
                # Setup GPIO for digital reading
                GPIO.setmode(GPIO.BCM)
                GPIO.setup([self.pin1, self.pin2], GPIO.IN)
                self.logger.info(f"Soil sensors using GPIO {self.pin1}, {self.pin2} (digital)")
        except Exception as e:
            self.logger.error(f"Soil sensor setup error: {e}")
    
    def read_adc_channel(self, channel):
        """Read from MCP3008 ADC channel"""
        if not self.spi:
            return 512  # Default value
        
        try:
            adc = self.spi.xfer2([1, (8 + channel) << 4, 0])
            data = ((adc[1] & 3) << 8) + adc[2]
            return data
        except Exception as e:
            self.logger.error(f"ADC reading error: {e}")
            return 512
    
    def read_single_sensor(self, pin, adc_channel, sensor_name="Sensor"):
        """Read individual soil sensor"""
        try:
            if self.use_adc:
                # Read analog value from ADC
                raw_value = self.read_adc_channel(adc_channel)
                
                # Convert to percentage (adjust these values for your sensor)
                dry_value = 700   # Adjust based on calibration
                wet_value = 300   # Adjust based on calibration
                
                # Calculate percentage (0% = dry, 100% = wet)
                moisture_percent = max(0, min(100, 
                    (dry_value - raw_value) / (dry_value - wet_value) * 100))
            else:
                # Digital reading (LOW = WET, HIGH = DRY)
                digital_value = GPIO.input(pin)
                raw_value = digital_value
                moisture_percent = 30.0 if digital_value == GPIO.HIGH else 70.0
            
            return {
                'moisture_level': round(moisture_percent, 1),
                'raw_value': raw_value,
                'status': 'WET' if moisture_percent > 50 else 'DRY'
            }
            
        except Exception as e:
            self.logger.error(f"{sensor_name} reading error: {e}")
            return {
                'moisture_level': 0.0,
                'raw_value': 0,
                'status': 'ERROR'
            }
    
    def read_moisture(self):
        """Read both soil moisture sensors"""
        try:
            # Read Sensor 1
            sensor1 = self.read_single_sensor(self.pin1, self.adc_channel1, "Soil1")
            
            # Read Sensor 2
            sensor2 = self.read_single_sensor(self.pin2, self.adc_channel2, "Soil2")
            
            # Calculate average
            avg_moisture = (sensor1['moisture_level'] + sensor2['moisture_level']) / 2
            
            # Determine overall status
            any_dry = (sensor1['status'] == 'DRY' or sensor2['status'] == 'DRY')
            both_dry = (sensor1['status'] == 'DRY' and sensor2['status'] == 'DRY')
            
            return {
                'soil1': sensor1['status'],
                'soil1_moisture': sensor1['moisture_level'],
                'soil1_raw': sensor1['raw_value'],
                
                'soil2': sensor2['status'],
                'soil2_moisture': sensor2['moisture_level'],
                'soil2_raw': sensor2['raw_value'],
                
                'average_moisture': round(avg_moisture, 1),
                'any_dry': any_dry,
                'both_dry': both_dry,
                
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Dual soil moisture reading error: {e}")
            return {
                'soil1': 'ERROR',
                'soil1_moisture': 0.0,
                'soil2': 'ERROR',
                'soil2_moisture': 0.0,
                'average_moisture': 0.0,
                'any_dry': False,
                'both_dry': False,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'error'
            }
    
    def calibrate_sensor(self, sensor_num=1):
        """Calibration helper - run in air and water"""
        if not self.use_adc:
            self.logger.info("Calibration only available with ADC")
            return
        
        channel = self.adc_channel1 if sensor_num == 1 else self.adc_channel2
        
        print(f"🔧 Soil Sensor {sensor_num} Calibration")
        print(f"1. Place sensor in DRY soil/air and press Enter")
        input()
        dry_reading = self.read_adc_channel(channel)
        print(f"Dry reading: {dry_reading}")
        
        print(f"2. Place sensor in WET soil/water and press Enter")
        input()
        wet_reading = self.read_adc_channel(channel)
        print(f"Wet reading: {wet_reading}")
        
        print(f"\nCalibration Results for Sensor {sensor_num}:")
        print(f"Update your config.py with:")
        print(f"SOIL_DRY_VALUE = {dry_reading}")
        print(f"SOIL_WET_VALUE = {wet_reading}")
    
    def cleanup(self):
        try:
            if self.spi:
                self.spi.close()
            GPIO.cleanup([self.pin1, self.pin2])
        except:
            pass
