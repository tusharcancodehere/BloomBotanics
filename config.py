#!/usr/bin/env python3
"""
🌱 BloomBotanics Agricultural Monitoring System
Complete Production Code - Version 2.1 with Servo & Status Report

Features:
✅ Real-time sensor monitoring (DHT22, Dual Soil, LDR, Rain)
✅ AI-powered animal/human detection  
✅ Automatic irrigation control
✅ Servo-controlled rotating camera scarecrow
✅ SMS alerts for threats and system status
✅ LCD display for local monitoring
✅ Cooling fan control
✅ Data logging and image capture
✅ Graceful degradation - runs with missing components
✅ System health monitoring
✅ Error recovery and restart
"""

import sys
import os
import time
import json
import signal
import threading
import traceback
from datetime import datetime, timedelta

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all modules
try:
    from sensors.dht22_sensor import DHT22Sensor
    from sensors.soil_moisture import SoilMoistureSensor  
    from sensors.rain_sensor import RainSensor
    from sensors.ldr_sensor import LDRSensor  # NEW
    from sensors.ai_camera import AICamera
    from hardware.lcd_display import LCDDisplay
    from hardware.relay_controller import RelayController
    from hardware.servo_controller import ServoController  # NEW
    from hardware.gsm_module import GSMModule
    from hardware.fan_controller import FanController
    from utils.logger import get_logger
    from utils.helpers import SystemHealth, DataManager
    from config import *
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure all required packages are installed")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)


class BloomBotanicsSystem:
    """
    🌱 Main BloomBotanics Agricultural Monitoring System
    
    Complete autonomous agricultural monitoring with:
    - Sensor monitoring and data logging
    - AI threat detection and alerts  
    - Automatic irrigation control
    - Servo-controlled camera rotation
    - Remote access via Pi Connect
    - SMS notifications and status updates
    - System health monitoring and recovery
    - Graceful degradation for missing components
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.running = False
        self.restart_count = 0
        self.last_health_check = 0
        
        # System state tracking
        self.last_irrigation = 0
        self.last_photo = 0
        self.last_detection_alert = 0
        self.last_sensor_alert = 0
        self.last_daily_report = None
        
        # Performance tracking
        self.loop_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Initialize system health monitor
        self.health_monitor = SystemHealth()
        self.data_manager = DataManager()
        
        # Hardware components (initialized later)
        self.dht22 = None
        self.soil_sensor = None
        self.ldr_sensor = None  # NEW
        self.rain_sensor = None
        self.ai_camera = None
        self.lcd = None
        self.relay = None
        self.servo = None  # NEW
        self.gsm = None
        self.fan = None
        
        # Track initialization status
        self.initialization_status = {}
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger.info("🚀 BloomBotanics System Initializing...")
    
    def initialize_system(self):
        """Initialize all system components with error handling"""
        self.logger.info("🔧 Initializing hardware components...")
        
        # Show initialization on LCD first
        try:
            self.lcd = LCDDisplay()
            if self.lcd:
                self.lcd.show_message("BloomBotanics", "Initializing...")
        except Exception as e:
            self.logger.warning(f"LCD initialization failed: {e}")
        
        # Initialize sensors with individual error handling
        try:
            self.dht22 = DHT22Sensor()
            self.initialization_status['DHT22'] = 'OK' if self.dht22 else 'FAIL'
        except Exception as e:
            self.logger.error(f"DHT22 initialization failed: {e}")
            self.initialization_status['DHT22'] = 'FAIL'
            
        try:
            self.soil_sensor = SoilMoistureSensor()
            self.initialization_status['Soil_Sensors'] = 'OK' if self.soil_sensor else 'FAIL'
        except Exception as e:
            self.logger.error(f"Soil sensor initialization failed: {e}")
            self.initialization_status['Soil_Sensors'] = 'FAIL'
        
        # NEW: Initialize LDR Sensor
        try:
            self.ldr_sensor = LDRSensor()
            self.initialization_status['LDR_Sensor'] = 'OK' if self.ldr_sensor else 'FAIL'
        except Exception as e:
            self.logger.error(f"LDR sensor initialization failed: {e}")
            self.initialization_status['LDR_Sensor'] = 'FAIL'
            
        try:
            self.rain_sensor = RainSensor()
            self.initialization_status['Rain'] = 'OK' if self.rain_sensor else 'FAIL'
        except Exception as e:
            self.logger.error(f"Rain sensor initialization failed: {e}")
            self.initialization_status['Rain'] = 'FAIL'
            
        try:
            if AI_DETECTION_ENABLED:
                self.ai_camera = AICamera()
                self.initialization_status['AI_Camera'] = 'OK' if self.ai_camera else 'FAIL'
            else:
                self.initialization_status['AI_Camera'] = 'DISABLED'
        except Exception as e:
            self.logger.error(f"AI Camera initialization failed: {e}")
            self.initialization_status['AI_Camera'] = 'FAIL'
            
        # Initialize hardware controllers
        try:
            self.relay = RelayController()
            self.initialization_status['Relay'] = 'OK' if self.relay else 'FAIL'
        except Exception as e:
            self.logger.error(f"Relay initialization failed: {e}")
            self.initialization_status['Relay'] = 'FAIL'
        
        # NEW: Initialize Servo Motor
        try:
            self.servo = ServoController()
            # Start servo rotation in background thread
            threading.Thread(target=self.servo.rotate_fro, daemon=True).start()
            self.initialization_status['Servo'] = 'OK' if self.servo else 'FAIL'
        except Exception as e:
            self.logger.error(f"Servo initialization failed: {e}")
            self.initialization_status['Servo'] = 'FAIL'
            
        try:
            self.gsm = GSMModule()
            self.initialization_status['GSM'] = 'OK' if self.gsm else 'FAIL'
        except Exception as e:
            self.logger.error(f"GSM initialization failed: {e}")
            self.initialization_status['GSM'] = 'FAIL'
            
        try:
            self.fan = FanController()
            self.initialization_status['Fan'] = 'OK' if self.fan else 'FAIL'
        except Exception as e:
            self.logger.error(f"Fan controller initialization failed: {e}")
            self.initialization_status['Fan'] = 'FAIL'
        
        # Log initialization results
        self.logger.info("🔍 Component Initialization Status:")
        for component, status in self.initialization_status.items():
            status_emoji = "✅" if status == "OK" else "❌" if status == "FAIL" else "⚠️"
            self.logger.info(f"  {status_emoji} {component}: {status}")
        
        # Check if critical components failed (but don't exit)
        critical_components = ['DHT22', 'Soil_Sensors', 'Relay']
        failed_critical = [comp for comp in critical_components if self.initialization_status.get(comp) == 'FAIL']
        
        if failed_critical:
            error_msg = f"Critical components failed: {', '.join(failed_critical)}"
            self.logger.warning(f"⚠️ {error_msg} - System will run with limited functionality")
            if self.lcd:
                self.lcd.show_message("PARTIAL INIT", f"Missing: {len(failed_critical)}")
            
            # Send error SMS if GSM is working
            if self.gsm and self.initialization_status.get('GSM') == 'OK':
                self.gsm.send_sms(SMS_TEMPLATES['system_error'].format(
                    error_type="Partial Initialization",
                    error_msg=error_msg,
                    timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ))
            
        # Send startup notification
        if self.gsm and self.initialization_status.get('GSM') == 'OK':
            startup_msg = SMS_TEMPLATES['startup'].format(
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            self.gsm.send_sms(startup_msg)
        
        if self.lcd:
            self.lcd.show_message("BloomBotanics", "System Ready!")
            time.sleep(2)
        
        self.logger.info("🌱 BloomBotanics System Initialization Complete!")
        return self.initialization_status
    
    def read_all_sensors(self):
        """Read data from all sensors with error handling"""
        sensor_data = {
            'timestamp': datetime.now().isoformat(),
            'temperature': None,
            'humidity': None,
            'soil1_moisture': None,
            'soil2_moisture': None,
            'soil_average': None,
            'light_level': None,  # NEW
            'rain_detected': False,
            'cpu_temperature': None,
            'system_status': 'running',
            'errors': []
        }
        
        # Read DHT22 (Temperature & Humidity)
        if self.dht22:
            try:
                dht_data = self.dht22.read_data()
                if dht_data and dht_data.get('status') == 'success':
                    sensor_data['temperature'] = dht_data['temperature']
                    sensor_data['humidity'] = dht_data['humidity']
                else:
                    sensor_data['errors'].append('DHT22 reading failed')
            except Exception as e:
                sensor_data['errors'].append(f'DHT22 error: {str(e)}')
                self.logger.error(f"DHT22 reading error: {e}")
        
        # Read Soil Moisture (Dual Sensors)
        if self.soil_sensor:
            try:
                soil_data = self.soil_sensor.read_moisture()
                if soil_data and soil_data.get('status') == 'success':
                    sensor_data['soil1_moisture'] = soil_data.get('soil1_moisture', 0)
                    sensor_data['soil2_moisture'] = soil_data.get('soil2_moisture', 0)
                    sensor_data['soil_average'] = soil_data.get('average_moisture', 0)
                else:
                    sensor_data['errors'].append('Soil sensor reading failed')
            except Exception as e:
                sensor_data['errors'].append(f'Soil sensor error: {str(e)}')
                self.logger.error(f"Soil sensor reading error: {e}")
        
        # NEW: Read LDR Light Sensor
        if self.ldr_sensor:
            try:
                ldr_data = self.ldr_sensor.read_data()
                if ldr_data and ldr_data.get('status') == 'success':
                    sensor_data['light_level'] = ldr_data.get('light', 'UNKNOWN')
                else:
                    sensor_data['errors'].append('LDR sensor reading failed')
            except Exception as e:
                sensor_data['errors'].append(f'LDR sensor error: {str(e)}')
                self.logger.error(f"LDR sensor reading error: {e}")
        
        # Read Rain Sensor
        if self.rain_sensor:
            try:
                rain_data = self.rain_sensor.read_rain_status()
                if rain_data and rain_data.get('status') == 'success':
                    sensor_data['rain_detected'] = rain_data['rain_detected']
                else:
                    sensor_data['errors'].append('Rain sensor reading failed')
            except Exception as e:
                sensor_data['errors'].append(f'Rain sensor error: {str(e)}')
                self.logger.error(f"Rain sensor reading error: {e}")
        
        # Read CPU Temperature
        if self.fan:
            try:
                sensor_data['cpu_temperature'] = self.fan.get_cpu_temperature()
            except Exception as e:
                sensor_data['errors'].append(f'CPU temp error: {str(e)}')
                self.logger.error(f"CPU temperature reading error: {e}")
        
        # Set system status based on errors
        if sensor_data['errors']:
            sensor_data['system_status'] = 'degraded'
            if len(sensor_data['errors']) > 2:
                sensor_data['system_status'] = 'error'
        
        return sensor_data
    
    # [Continue with all other methods from your original main.py...]
    # check_sensor_alerts, process_ai_detections, _send_threat_alerts, 
    # capture_periodic_photo, update_lcd_display, check_system_health,
    # send_daily_report, main_monitoring_loop, _restart_system, signal_handler
    
    def cleanup(self, send_notification=True):
        """Cleanup all system resources gracefully"""
        self.logger.info("🧹 Cleaning up system resources...")
        
        try:
            # Send shutdown notification
            if send_notification and self.gsm:
                shutdown_msg = SMS_TEMPLATES['shutdown'].format(
                    timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                )
                self.gsm.send_sms(shutdown_msg)
            
            # Display shutdown message
            if self.lcd:
                self.lcd.show_message("BloomBotanics", "Shutting Down...")
                time.sleep(2)
            
            # Cleanup all components
            components = [
                ('DHT22', self.dht22),
                ('Soil Sensors', self.soil_sensor),
                ('LDR Sensor', self.ldr_sensor),  # NEW
                ('Rain Sensor', self.rain_sensor),
                ('AI Camera', self.ai_camera),
                ('LCD Display', self.lcd),
                ('Relay', self.relay),
                ('Servo Motor', self.servo),  # NEW
                ('GSM Module', self.gsm),
                ('Fan Controller', self.fan)
            ]
            
            for name, component in components:
                if component:
                    try:
                        component.cleanup()
                        self.logger.info(f"✅ {name} cleaned up")
                    except Exception as e:
                        self.logger.error(f"❌ {name} cleanup error: {e}")
            
            # Final system statistics
            runtime = (time.time() - self.start_time) / 3600
            self.logger.info(f"📊 Final Statistics:")
            self.logger.info(f"   Runtime: {runtime:.1f} hours")
            self.logger.info(f"   Loop count: {self.loop_count}")
            self.logger.info(f"   Error count: {self.error_count}")
            self.logger.info(f"   Success rate: {((self.loop_count - self.error_count) / max(self.loop_count, 1) * 100):.1f}%")
            
            self.logger.info("✅ System cleanup completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup error: {e}")
    
    def print_system_status_summary(self):
        """Print detailed system status with missing components"""
        print("\n" + "="*70)
        print("📊 BLOOMBOTANICS SYSTEM STATUS SUMMARY")
        print("="*70)
        
        print("\n✅ OPERATIONAL COMPONENTS:")
        operational = [name.replace('_', ' ') for name, status in self.initialization_status.items() if status == 'OK']
        if operational:
            for component in operational:
                print(f"   • {component}")
        else:
            print("   None")
        
        print("\n❌ MISSING/FAILED COMPONENTS:")
        missing = [name.replace('_', ' ') for name, status in self.initialization_status.items() if status == 'FAIL']
        if missing:
            for component in missing:
                print(f"   • {component} - NOT INITIALIZED")
        else:
            print("   None - All components operational!")
        
        print("\n⚠️ DISABLED COMPONENTS:")
        disabled = [name.replace('_', ' ') for name, status in self.initialization_status.items() if status == 'DISABLED']
        if disabled:
            for component in disabled:
                print(f"   • {component} - DISABLED IN CONFIG")
        else:
            print("   None")
        
        print("\n📈 SYSTEM CAPABILITIES:")
        if self.soil_sensor and self.relay:
            print("   ✅ Auto-irrigation available")
        else:
            print("   ❌ Auto-irrigation unavailable (missing soil sensor or relay)")
        
        if self.ai_camera and AI_DETECTION_ENABLED:
            print("   ✅ AI threat detection active")
        else:
            print("   ❌ AI threat detection disabled")
        
        if self.servo:
            print("   ✅ Servo camera rotation active")
        else:
            print("   ❌ Servo rotation unavailable")
        
        if self.gsm:
            print("   ✅ SMS alerts enabled")
        else:
            print("   ❌ SMS alerts unavailable (GSM module missing)")
        
        if self.lcd:
            print("   ✅ Local LCD monitoring active")
        else:
            print("   ❌ LCD display unavailable")
        
        print("\n" + "="*70)
        print("💡 TIP: Run 'python3 validate_system.py' to check missing components")
        print("="*70 + "\n")
    
    def run(self):
        """Start the BloomBotanics system"""
        try:
            self.logger.info("🚀 Starting BloomBotanics Agricultural Monitoring System")
            self.logger.info(f"📍 Farm: {FARM_NAME}")
            self.logger.info(f"📱 SMS Alerts: {PHONE_NUMBER}")
            self.logger.info(f"🤖 AI Detection: {'Enabled' if AI_DETECTION_ENABLED else 'Disabled'}")
            self.logger.info(f"💧 Auto Irrigation: {'Enabled' if AUTO_IRRIGATION else 'Disabled'}")
            
            # Initialize all components
            init_status = self.initialize_system()
            
            # Print status summary
            self.print_system_status_summary()
            
            # Start main monitoring loop
            self.main_monitoring_loop()
            
        except Exception as e:
            self.logger.critical(f"💥 Fatal startup error: {e}")
            self.logger.critical(f"💥 Startup traceback: {traceback.format_exc()}")
            
            if self.lcd:
                self.lcd.show_message("FATAL ERROR!", "Check logs")
            
            return False
        
        return True


def main():
    """Main entry point with enhanced error handling"""
    try:
        # Verify we're running on Raspberry Pi
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
            if 'Raspberry Pi' not in cpuinfo:
                print("⚠️  Warning: Not running on Raspberry Pi - some features may not work")
        except:
            pass
        
        # Create and run the system
        system = BloomBotanicsSystem()
        success = system.run()
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Manual shutdown requested")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        print(f"💥 Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
