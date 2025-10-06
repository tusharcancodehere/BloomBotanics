#!/usr/bin/env python3
"""
BloomBotanics System Validator
Checks all dependencies, hardware, and displays missing components
"""

import sys
import os
import importlib
from datetime import datetime

class SystemValidator:
    def __init__(self):
        self.missing_packages = []
        self.missing_hardware = []
        self.missing_files = []
        self.warnings = []
        
    def check_python_packages(self):
        """Check required Python packages"""
        print("🐍 Checking Python packages...")
        
        required_packages = {
            'RPi.GPIO': 'RPi.GPIO',
            'Adafruit_DHT': 'Adafruit_DHT',
            'RPLCD': 'RPLCD',
            'picamera2': 'picamera2',
            'cv2': 'opencv-python',
            'numpy': 'numpy',
            'torch': 'torch',
            'serial': 'pyserial',
        }
        
        for module, package in required_packages.items():
            try:
                importlib.import_module(module)
                print(f"  ✅ {package}")
            except ImportError:
                self.missing_packages.append(package)
                print(f"  ❌ {package} - MISSING")
    
    def check_hardware(self):
        """Check hardware connections"""
        print("\n🔧 Checking hardware...")
        
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            
            # Test GPIO pins
            test_pins = {
                4: 'DHT22',
                22: 'Soil Sensor 1',
                23: 'Soil Sensor 2',
                24: 'LDR Sensor',
                27: 'Rain Sensor',
                18: 'Relay (Pump)',
                17: 'Servo Motor'
            }
            
            for pin, name in test_pins.items():
                try:
                    GPIO.setup(pin, GPIO.OUT)
                    print(f"  ✅ GPIO {pin} ({name})")
                except:
                    self.missing_hardware.append(f"GPIO {pin} ({name})")
                    print(f"  ⚠️  GPIO {pin} ({name}) - Not configured")
            
            GPIO.cleanup()
            
        except Exception as e:
            self.warnings.append(f"GPIO check failed: {e}")
            print(f"  ⚠️  GPIO check failed: {e}")
        
        # Check I2C LCD
        try:
            import smbus
            bus = smbus.SMBus(1)
            bus.write_byte(0x27, 0)
            print("  ✅ LCD Display (I2C 0x27)")
        except:
            self.missing_hardware.append("LCD Display (I2C 0x27)")
            print("  ❌ LCD Display (I2C 0x27) - NOT FOUND")
        
        # Check Camera
        try:
            from picamera2 import Picamera2
            print("  ✅ Pi Camera")
        except:
            self.missing_hardware.append("Pi Camera")
            print("  ⚠️  Pi Camera - Not detected")
    
    def check_project_files(self):
        """Check project structure"""
        print("\n📁 Checking project files...")
        
        required_files = [
            'main.py',
            'config.py',
            'requirements.txt',
            'sensors/__init__.py',
            'sensors/dht22_sensor.py',
            'sensors/soil_moisture.py',
            'sensors/ldr_sensor.py',
            'sensors/rain_sensor.py',
            'sensors/ai_camera.py',
            'hardware/__init__.py',
            'hardware/lcd_display.py',
            'hardware/relay_controller.py',
            'hardware/servo_controller.py',
            'hardware/gsm_module.py',
            'hardware/fan_controller.py',
            'utils/__init__.py',
            'utils/logger.py',
            'utils/helpers.py',
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"  ✅ {file_path}")
            else:
                self.missing_files.append(file_path)
                print(f"  ❌ {file_path} - MISSING")
    
    def generate_report(self):
        """Generate validation report"""
        print("\n" + "="*60)
        print("📊 VALIDATION REPORT")
        print("="*60)
        
        if not self.missing_packages and not self.missing_hardware and not self.missing_files:
            print("✅ ALL CHECKS PASSED!")
            print("🎉 System is ready to run!")
            return True
        
        if self.missing_packages:
            print(f"\n❌ MISSING PYTHON PACKAGES ({len(self.missing_packages)}):")
            for pkg in self.missing_packages:
                print(f"   - {pkg}")
            print(f"   Install with: pip install {' '.join(self.missing_packages)}")
        
        if self.missing_hardware:
            print(f"\n⚠️  HARDWARE ISSUES ({len(self.missing_hardware)}):")
            for hw in self.missing_hardware:
                print(f"   - {hw}")
            print("   Check wiring and connections")
        
        if self.missing_files:
            print(f"\n❌ MISSING FILES ({len(self.missing_files)}):")
            for file in self.missing_files:
                print(f"   - {file}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        print("\n" + "="*60)
        return False

def main():
    print("🌱 BloomBotanics System Validator")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    validator = SystemValidator()
    validator.check_python_packages()
    validator.check_hardware()
    validator.check_project_files()
    
    success = validator.generate_report()
    
    if not success:
        print("\n⚠️  System validation failed. Please fix the issues above.")
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()
