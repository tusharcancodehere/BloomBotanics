#!/usr/bin/env python3
"""
BloomBotanics Sensor Test Script
Individual testing of all sensors and hardware components
"""

import sys
import os
import time
import json
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import *
    from sensors.dht22_sensor import DHT22Sensor
    from sensors.soil_moisture import SoilMoistureSensor
    from sensors.rain_sensor import RainSensor
    from sensors.ai_camera import AICamera
    from hardware.lcd_display import LCDDisplay
    from hardware.relay_controller import RelayController
    from hardware.gsm_module import GSMModule
    from hardware.fan_controller import FanController
    from utils.logger import get_logger
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running this from the BloomBotanics directory")
    sys.exit(1)

class SensorTester:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.test_results = {}
        self.start_time = time.time()
        
    def log_test_result(self, component, success, details="", error=None):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {component}")
        
        if details:
            print(f"     {details}")
        
        if error:
            print(f"     Error: {error}")
        
        self.test_results[component] = {
            'success': success,
            'details': details,
            'error': str(error) if error else None,
            'timestamp': datetime.now().isoformat()
        }
        
        print()
    
    def test_dht22_sensor(self):
        """Test DHT22 temperature/humidity sensor"""
        print("🌡️ Testing DHT22 Temperature/Humidity Sensor...")
        
        try:
            dht = DHT22Sensor()
            
            if not dht.sensor:
                self.log_test_result("DHT22", False, error="Sensor not initialized")
                return False
            
            # Take multiple readings
            successful_readings = 0
            temperatures = []
            humidities = []
            
            for i in range(3):
                print(f"   Reading {i+1}/3...", end=" ", flush=True)
                data = dht.read_data()
                
                if data and data.get('status') == 'success':
                    successful_readings += 1
                    temperatures.append(data['temperature'])
                    humidities.append(data['humidity'])
                    print(f"T: {data['temperature']}°C, H: {data['humidity']}%")
                else:
                    print("Failed")
                
                time.sleep(2)
            
            dht.cleanup()
            
            if successful_readings >= 2:
                avg_temp = sum(temperatures) / len(temperatures)
                avg_humidity = sum(humidities) / len(humidities)
                details = f"Avg Temp: {avg_temp:.1f}°C, Avg Humidity: {avg_humidity:.1f}% ({successful_readings}/3 readings)"
                self.log_test_result("DHT22", True, details)
                return True
            else:
                self.log_test_result("DHT22", False, error=f"Only {successful_readings}/3 readings successful")
                return False
                
        except Exception as e:
            self.log_test_result("DHT22", False, error=e)
            return False
    
    def test_soil_sensor(self):
        """Test soil moisture sensor"""
        print("🌱 Testing Soil Moisture Sensor...")
        
        try:
            soil = SoilMoistureSensor()
            
            # Take multiple readings
            readings = []
            for i in range(3):
                print(f"   Reading {i+1}/3...", end=" ", flush=True)
                data = soil.read_moisture()
                
                if data and data.get('status') == 'success':
                    readings.append(data['moisture_level'])
                    print(f"Moisture: {data['moisture_level']:.1f}% (Raw: {data.get('raw_value', 'N/A')})")
                else:
                    print("Failed")
                
                time.sleep(1)
            
            soil.cleanup()
            
            if len(readings) >= 2:
                avg_moisture = sum(readings) / len(readings)
                details = f"Avg Moisture: {avg_moisture:.1f}% ({len(readings)}/3 readings)"
                
                # Check if readings are reasonable
                if 0 <= avg_moisture <= 100:
                    self.log_test_result("Soil Moisture", True, details)
                    return True
                else:
                    self.log_test_result("Soil Moisture", False, error=f"Unreasonable readings: {avg_moisture:.1f}%")
                    return False
            else:
                self.log_test_result("Soil Moisture", False, error="Insufficient successful readings")
                return False
                
        except Exception as e:
            self.log_test_result("Soil Moisture", False, error=e)
            return False
    
    def test_rain_sensor(self):
        """Test rain detection sensor"""
        print("☔ Testing Rain Detection Sensor...")
        
        try:
            rain = RainSensor()
            
            # Take multiple readings
            readings = []
            for i in range(3):
                print(f"   Reading {i+1}/3...", end=" ", flush=True)
                data = rain.read_rain_status()
                
                if data and data.get('status') == 'success':
                    readings.append(data)
                    rain_status = "RAIN" if data['rain_detected'] else "DRY"
                    print(f"{rain_status} (Intensity: {data.get('intensity', 'N/A')})")
                else:
                    print("Failed")
                
                time.sleep(1)
            
            rain.cleanup()
            
            if len(readings) >= 2:
                rain_detections = sum(1 for r in readings if r['rain_detected'])
                details = f"Rain detected in {rain_detections}/3 readings"
                self.log_test_result("Rain Sensor", True, details)
                return True
            else:
                self.log_test_result("Rain Sensor", False, error="Insufficient successful readings")
                return False
                
        except Exception as e:
            self.log_test_result("Rain Sensor", False, error=e)
            return False
    
    def test_camera(self):
        """Test camera module"""
        print("📸 Testing Camera Module...")
        
        try:
            camera = AICamera()
            
            if not camera.camera:
                self.log_test_result("Camera", False, error="Camera not initialized")
                return False
            
            # Test basic photo capture
            print("   Capturing test photo...", end=" ", flush=True)
            test_photo = camera.capture_photo("/tmp/bloom_test_photo.jpg")
            
            if test_photo and os.path.exists(test_photo):
                print("Success")
                
                # Check file size (should be > 10KB for a valid image)
                file_size = os.path.getsize(test_photo)
                if file_size > 10000:
                    details = f"Photo captured successfully ({file_size} bytes)"
                    
                    # Test AI detection if enabled
                    if AI_DETECTION_ENABLED and camera.model:
                        print("   Testing AI detection...", end=" ", flush=True)
                        detections, frame = camera.detect_threats()
                        print(f"Found {len(detections)} objects")
                        details += f", AI detection: {len(detections)} objects found"
                    
                    # Cleanup test photo
                    try:
                        os.remove(test_photo)
                    except:
                        pass
                    
                    camera.cleanup()
                    self.log_test_result("Camera", True, details)
                    return True
                else:
                    camera.cleanup()
                    self.log_test_result("Camera", False, error=f"Invalid image file size: {file_size} bytes")
                    return False
            else:
                print("Failed")
                camera.cleanup()
                self.log_test_result("Camera", False, error="Failed to capture photo")
                return False
                
        except Exception as e:
            self.log_test_result("Camera", False, error=e)
            return False
    
    def test_lcd_display(self):
        """Test LCD display"""
        print("💻 Testing LCD Display...")
        
        try:
            lcd = LCDDisplay()
            
            if not lcd.lcd:
                self.log_test_result("LCD Display", False, error="LCD not initialized")
                return False
            
            # Test basic display
            print("   Displaying test message...", end=" ", flush=True)
            lcd.show_message("BloomBotanics", "Sensor Test")
            time.sleep(2)
            print("Success")
            
            # Test sensor data display
            print("   Testing sensor data display...", end=" ", flush=True)
            lcd.display_sensor_data(25.5, 65.0, 45.0, False)
            time.sleep(2)
            print("Success")
            
            # Test alert display
            print("   Testing alert display...", end=" ", flush=True)
            lcd.show_threat_alert("Test Alert", 1)
            print("Success")
            
            lcd.cleanup()
            self.log_test_result("LCD Display", True, "All display tests passed")
            return True
            
        except Exception as e:
            self.log_test_result("LCD Display", False, error=e)
            return False
    
    def test_relay_controller(self):
        """Test relay controller (irrigation)"""
        print("💧 Testing Relay Controller...")
        
        try:
            relay = RelayController()
            
            # Test basic on/off
            print("   Testing relay ON...", end=" ", flush=True)
            success_on = relay.turn_on()
            time.sleep(1)
            print("Success" if success_on else "Failed")
            
            print("   Testing relay OFF...", end=" ", flush=True)
            success_off = relay.turn_off()
            time.sleep(1)
            print("Success" if success_off else "Failed")
            
            # Test timed activation
            print("   Testing 3-second timed activation...", end=" ", flush=True)
            success_timed = relay.activate(3)
            if success_timed:
                time.sleep(4)  # Wait for activation to complete
                print("Success")
            else:
                print("Failed")
            
            # Test emergency stop
            print("   Testing emergency stop...", end=" ", flush=True)
            relay.activate(10)  # Start long activation
            time.sleep(1)
            success_stop = relay.emergency_stop()
            print("Success" if success_stop else "Failed")
            
            # Get status
            status = relay.get_status()
            details = f"Status: {status.get('status', 'unknown')}, Activations: {status.get('activation_count', 0)}"
            
            relay.cleanup()
            
            if success_on and success_off and success_timed and success_stop:
                self.log_test_result("Relay Controller", True, details)
                return True
            else:
                self.log_test_result("Relay Controller", False, error="One or more relay tests failed")
                return False
                
        except Exception as e:
            self.log_test_result("Relay Controller", False, error=e)
            return False
    
    def test_gsm_module(self):
        """Test GSM module"""
        print("📱 Testing GSM Module...")
        
        try:
            gsm = GSMModule()
            
            if not gsm.connected:
                self.log_test_result("GSM Module", False, error="GSM not connected")
                return False
            
            # Test basic AT command
            print("   Testing basic communication...", end=" ", flush=True)
            if gsm.send_at_command("AT", "OK"):
                print("Success")
            else:
                print("Failed")
                gsm.cleanup()
                self.log_test_result("GSM Module", False, error="Basic AT command failed")
                return False
            
            # Get status
            status = gsm.get_status()
            print(f"   Signal strength: {status.get('signal_strength', 'Unknown')}")
            print(f"   Network status: {status.get('network_status', 'Unknown')}")
            
            # Test SMS (optional - requires user confirmation)
            print("   Send test SMS? (y/N): ", end="")
            try:
                import select
                import sys
                
                # Non-blocking input with timeout
                ready, _, _ = select.select([sys.stdin], [], [], 5)
                if ready:
                    response = sys.stdin.readline().strip().lower()
                    if response == 'y':
                        print("   Sending test SMS...", end=" ", flush=True)
                        test_msg = f"BloomBotanics Test SMS - {datetime.now().strftime('%H:%M:%S')}"
                        if gsm.send_sms(test_msg):
                            print("Success")
                            sms_test = True
                        else:
                            print("Failed")
                            sms_test = False
                    else:
                        print("   SMS test skipped")
                        sms_test = True  # Don't fail test if user skipped
                else:
                    print("   SMS test skipped (timeout)")
                    sms_test = True
            except:
                print("   SMS test skipped")
                sms_test = True
            
            gsm.cleanup()
            
            details = f"Signal: {status.get('signal_strength', 'N/A')}, Network: {status.get('network_status', 'N/A')}"
            self.log_test_result("GSM Module", sms_test, details)
            return sms_test
            
        except Exception as e:
            self.log_test_result("GSM Module", False, error=e)
            return False
    
    def test_fan_controller(self):
        """Test fan controller"""
        print("🌀 Testing Fan Controller...")
        
        try:
            fan = FanController()
            
            if not fan.pwm:
                self.log_test_result("Fan Controller", False, error="PWM not initialized")
                return False
            
            # Test different speeds
            test_speeds = [0, 25, 50, 75, 100, 0]
            speed_tests = []
            
            for speed in test_speeds:
                print(f"   Setting fan speed to {speed}%...", end=" ", flush=True)
                success = fan.set_speed(speed)
                speed_tests.append(success)
                time.sleep(1)
                print("Success" if success else "Failed")
            
            # Test temperature reading
            print("   Reading CPU temperature...", end=" ", flush=True)
            cpu_temp = fan.get_cpu_temperature()
            print(f"{cpu_temp:.1f}°C")
            
            # Test auto control
            print("   Testing auto control...", end=" ", flush=True)
            fan.auto_control()
            status = fan.get_status()
            print(f"Speed set to {status.get('current_speed', 0)}%")
            
            fan.cleanup()
            
            if all(speed_tests):
                details = f"All speed tests passed, CPU temp: {cpu_temp:.1f}°C"
                self.log_test_result("Fan Controller", True, details)
                return True
            else:
                failed_speeds = sum(1 for test in speed_tests if not test)
                self.log_test_result("Fan Controller", False, error=f"{failed_speeds} speed tests failed")
                return False
                
        except Exception as e:
            self.log_test_result("Fan Controller", False, error=e)
            return False
    
    def test_system_health(self):
        """Test system health monitoring"""
        print("🏥 Testing System Health...")
        
        try:
            # Import system health monitor
            from utils.helpers import SystemHealth
            health = SystemHealth()
            
            print("   Checking system health...", end=" ", flush=True)
            health_data = health.get_system_health()
            
            if 'error' not in health_data:
                print("Success")
                
                # Display key metrics
                print(f"   CPU Temperature: {health_data.get('cpu_temp', 'N/A')}°C")
                print(f"   Memory Usage: {health_data.get('memory_usage', 'N/A')}%")
                print(f"   Disk Usage: {health_data.get('disk_usage', 'N/A')}%")
                print(f"   Network Connected: {health_data.get('network_connected', 'N/A')}")
                
                details = f"CPU: {health_data.get('cpu_temp', 'N/A')}°C, RAM: {health_data.get('memory_usage', 'N/A')}%, Disk: {health_data.get('disk_usage', 'N/A')}%"
                self.log_test_result("System Health", True, details)
                return True
            else:
                print("Failed")
                self.log_test_result("System Health", False, error=health_data.get('error'))
                return False
                
        except Exception as e:
            self.log_test_result("System Health", False, error=e)
            return False
    
    def test_configuration(self):
        """Test system configuration"""
        print("⚙️ Testing System Configuration...")
        
        try:
            config_issues = []
            
            # Check phone number
            if PHONE_NUMBER == "+919876543210":
                config_issues.append("Phone number not updated from default")
            
            # Check GPIO pins
            gpio_pins = [DHT22_DATA_PIN, SOIL_MOISTURE_PIN, RAIN_DIGITAL_PIN, RELAY_PIN, FAN_CONTROL_PIN]
            unique_pins = set(gpio_pins)
            if len(unique_pins) != len(gpio_pins):
                config_issues.append("Duplicate GPIO pins detected")
            
            # Check directories
            required_dirs = [SENSOR_DATA_DIR, IMAGE_DIR, DETECTION_DIR, LOG_DIR]
            for directory in required_dirs:
                if not os.path.exists(directory):
                    config_issues.append(f"Missing directory: {directory}")
            
            # Check AI settings
            if AI_DETECTION_ENABLED:
                if DETECTION_CONFIDENCE < 0.3 or DETECTION_CONFIDENCE > 0.9:
                    config_issues.append("AI detection confidence out of recommended range (0.3-0.9)")
            
            # Check thresholds
            if SOIL_MOISTURE_MIN >= SOIL_MOISTURE_MAX:
                config_issues.append("Invalid soil moisture thresholds")
            
            if TEMP_MIN >= TEMP_MAX:
                config_issues.append("Invalid temperature thresholds")
            
            if config_issues:
                issues_text = "; ".join(config_issues[:3])  # Limit to first 3 issues
                self.log_test_result("Configuration", False, error=issues_text)
                return False
            else:
                details = f"Phone: {PHONE_NUMBER[:7]}..., AI: {'Enabled' if AI_DETECTION_ENABLED else 'Disabled'}"
                self.log_test_result("Configuration", True, details)
                return True
                
        except Exception as e:
            self.log_test_result("Configuration", False, error=e)
            return False
    
    def run_all_tests(self):
        """Run all hardware tests"""
        print("🌱 BloomBotanics Hardware Test Suite")
        print("====================================")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # List of tests to run
        tests = [
            ("Configuration", self.test_configuration),
            ("System Health", self.test_system_health),
            ("DHT22 Sensor", self.test_dht22_sensor),
            ("Soil Moisture", self.test_soil_sensor),
            ("Rain Sensor", self.test_rain_sensor),
            ("Camera", self.test_camera),
            ("LCD Display", self.test_lcd_display),
            ("Relay Controller", self.test_relay_controller),
            ("Fan Controller", self.test_fan_controller),
            ("GSM Module", self.test_gsm_module),
        ]
        
        # Run tests
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_function in tests:
            try:
                if test_function():
                    passed_tests += 1
            except KeyboardInterrupt:
                print("\n🛑 Tests interrupted by user")
                break
            except Exception as e:
                print(f"❌ FAIL {test_name}")
                print(f"     Unexpected error: {e}")
                print()
        
        # Generate test report
        self.generate_test_report(passed_tests, total_tests)
    
    def run_specific_test(self, test_name):
        """Run a specific test"""
        test_map = {
            'config': self.test_configuration,
            'health': self.test_system_health,
            'dht22': self.test_dht22_sensor,
            'soil': self.test_soil_sensor,
            'rain': self.test_rain_sensor,
            'camera': self.test_camera,
            'lcd': self.test_lcd_display,
            'relay': self.test_relay_controller,
            'fan': self.test_fan_controller,
            'gsm': self.test_gsm_module,
        }
        
        if test_name.lower() in test_map:
            print(f"🔧 Running {test_name.upper()} Test")
            print("=" * 30)
            test_function = test_map[test_name.lower()]
            success = test_function()
            
            if success:
                print(f"✅ {test_name.upper()} test completed successfully")
            else:
                print(f"❌ {test_name.upper()} test failed")
            
            return success
        else:
            print(f"❌ Unknown test: {test_name}")
            print(f"Available tests: {', '.join(test_map.keys())}")
            return False
    
    def generate_test_report(self, passed, total):
        """Generate comprehensive test report"""
        runtime = time.time() - self.start_time
        
        print()
        print("📊 Test Summary")
        print("===============")
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        print(f"Total Runtime: {runtime:.1f} seconds")
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Detailed results
        print("📋 Detailed Results:")
        print("===================")
        
        for component, result in self.test_results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"{status} {component}")
            
            if result['details']:
                print(f"     {result['details']}")
            
            if result['error']:
                print(f"     Error: {result['error']}")
            
            print()
        
        # Save report to file
        try:
            report_data = {
                'test_summary': {
                    'passed': passed,
                    'total': total,
                    'success_rate': (passed/total)*100,
                    'runtime_seconds': runtime,
                    'timestamp': datetime.now().isoformat()
                },
                'test_results': self.test_results,
                'system_info': {
                    'python_version': sys.version,
                    'platform': os.uname()._asdict() if hasattr(os.uname(), '_asdict') else str(os.uname()),
                    'working_directory': os.getcwd()
                }
            }
            
            # Ensure logs directory exists
            os.makedirs(LOG_DIR, exist_ok=True)
            
            report_file = os.path.join(LOG_DIR, f'sensor_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            print(f"📄 Detailed report saved to: {report_file}")
            
        except Exception as e:
            print(f"❌ Could not save report: {e}")
        
        # Recommendations
        print()
        print("💡 Recommendations:")
        print("==================")
        
        failed_tests = [comp for comp, result in self.test_results.items() if not result['success']]
        
        if not failed_tests:
            print("🎉 All tests passed! Your BloomBotanics system is ready for deployment.")
        else:
            print("🔧 The following components need attention:")
            for component in failed_tests:
                error = self.test_results[component]['error']
                print(f"  - {component}: {error}")
            
            print()
            print("🛠️ Troubleshooting steps:")
            print("  1. Check wiring connections")
            print("  2. Verify GPIO pin assignments in config.py")
            print("  3. Ensure all required packages are installed")
            print("  4. Check hardware component power requirements")
            print("  5. Review system logs: sudo journalctl -u bloom-botanics -f")
        
        print()
        
        # Return overall success
        return passed == total

def main():
    """Main function"""
    tester = SensorTester()
    
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        success = tester.run_specific_test(test_name)
        sys.exit(0 if success else 1)
    else:
        # Run all tests
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)
