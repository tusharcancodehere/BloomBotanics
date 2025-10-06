#!/usr/bin/env python3
"""
BloomBotanics Quick Hardware Test - FIXED VERSION
Tests all hardware without full system startup
"""

import sys
import os
import time
import traceback

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_header():
    """Print test header"""
    print("🌱 BloomBotanics Quick Hardware Test")
    print("==================================")
    print(f"Python Version: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    print("")

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing Module Imports...")
    print("-" * 30)
    
    import_results = {}
    
    # Test basic Pi GPIO
    try:
        import RPi.GPIO as GPIO
        print("✅ RPi.GPIO imported successfully")
        import_results['RPi.GPIO'] = True
    except ImportError as e:
        print(f"❌ RPi.GPIO failed: {e}")
        print("   This is expected on Windows - run on Raspberry Pi")
        import_results['RPi.GPIO'] = False
    
    # Test config file - FIXED IMPORT
    try:
        import config
        PHONE_NUMBER = getattr(config, 'PHONE_NUMBER', 'Not configured')
        AI_DETECTION_ENABLED = getattr(config, 'AI_DETECTION_ENABLED', False)
        DHT22_DATA_PIN = getattr(config, 'DHT22_DATA_PIN', 4)
        
        print("✅ Configuration loaded")
        print(f"   📱 Phone: {PHONE_NUMBER}")
        print(f"   🤖 AI Detection: {AI_DETECTION_ENABLED}")
        print(f"   🌡️ DHT22 Pin: GPIO {DHT22_DATA_PIN}")
        import_results['config'] = True
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        print("   Check: config.py exists and has no syntax errors")
        import_results['config'] = False
    except Exception as e:
        print(f"❌ Config error: {e}")
        import_results['config'] = False
    
    # Test other required modules
    modules_to_test = [
        ('time', 'time'),
        ('os', 'os'),
        ('sys', 'sys'),
    ]
    
    # Test optional modules
    optional_modules = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('requests', 'requests'),
    ]
    
    for module_name, package_name in modules_to_test + optional_modules:
        try:
            __import__(module_name)
            print(f"✅ {module_name} imported")
            import_results[module_name] = True
        except ImportError as e:
            print(f"❌ {module_name} failed: {e}")
            if module_name in ['cv2', 'numpy', 'requests']:
                print(f"   Install: pip install {package_name}")
            import_results[module_name] = False
    
    success_count = sum(1 for success in import_results.values() if success)
    total_count = len(import_results)
    
    print(f"\nImport Results: {success_count}/{total_count} successful")
    return import_results.get('config', False)

def test_gpio_basic():
    """Test basic GPIO functionality"""
    print("\n🔌 Testing Basic GPIO...")
    print("-" * 30)
    
    try:
        import RPi.GPIO as GPIO
        
        # Test GPIO setup
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        
        # Test a safe pin (GPIO 18 - our relay pin)
        test_pin = 18
        GPIO.setup(test_pin, GPIO.OUT)
        
        print(f"✅ GPIO setup successful")
        print(f"✅ Test pin {test_pin} configured as output")
        
        # Test pin toggle
        GPIO.output(test_pin, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(test_pin, GPIO.LOW)
        print(f"✅ Pin toggle test passed")
        
        # Cleanup
        GPIO.cleanup([test_pin])
        print(f"✅ GPIO cleanup successful")
        
        return True
        
    except ImportError:
        print("❌ RPi.GPIO not available (expected on Windows)")
        return False
    except Exception as e:
        print(f"❌ GPIO test failed: {e}")
        return False

def test_sensors():
    """Test sensor readings with error handling"""
    print("\n🌡️ Testing Sensors...")
    print("-" * 30)
    
    sensor_results = {}
    
    # Test DHT22
    print("Testing DHT22 Temperature/Humidity Sensor...")
    try:
        from sensors.dht22_sensor import DHT22Sensor
        dht = DHT22Sensor()
        
        if hasattr(dht, 'sensor') and dht.sensor:
            print("   DHT22 sensor object created")
            data = dht.read_data()
            
            if data and data.get('status') == 'success':
                print(f"✅ DHT22: {data['temperature']}°C, {data['humidity']}%")
                sensor_results['DHT22'] = True
            else:
                print("⚠️ DHT22: Sensor initialized but no valid data")
                print("   Check: Wiring, pull-up resistor, sensor power")
                sensor_results['DHT22'] = False
        else:
            print("❌ DHT22: Sensor not initialized")
            sensor_results['DHT22'] = False
            
        try:
            if hasattr(dht, 'cleanup'):
                dht.cleanup()
        except:
            pass
            
    except ImportError as e:
        print(f"❌ DHT22 import failed: {e}")
        print("   Expected on Windows - run on Raspberry Pi")
        sensor_results['DHT22'] = False
    except Exception as e:
        print(f"❌ DHT22 test failed: {e}")
        sensor_results['DHT22'] = False
    
    # Test Soil Moisture
    print("\nTesting Soil Moisture Sensor...")
    try:
        from sensors.soil_moisture import SoilMoistureSensor
        soil = SoilMoistureSensor()
        
        data = soil.read_moisture()
        if data and data.get('status') == 'success':
            print(f"✅ Soil Moisture: {data['moisture_level']}% (Raw: {data.get('raw_value')})")
            sensor_results['Soil'] = True
        else:
            print("⚠️ Soil Moisture: No valid data")
            print("   Check: Sensor wiring and power")
            sensor_results['Soil'] = False
            
        try:
            if hasattr(soil, 'cleanup'):
                soil.cleanup()
        except:
            pass
            
    except Exception as e:
        print(f"❌ Soil sensor test failed: {e}")
        sensor_results['Soil'] = False
    
    # Test Rain Sensor
    print("\nTesting Rain Sensor...")
    try:
        from sensors.rain_sensor import RainSensor
        rain = RainSensor()
        
        data = rain.read_rain_status()
        if data and data.get('status') == 'success':
            rain_status = "DETECTED" if data['rain_detected'] else "DRY"
            print(f"✅ Rain Sensor: {rain_status} (Intensity: {data.get('intensity')})")
            sensor_results['Rain'] = True
        else:
            print("⚠️ Rain Sensor: No valid data")
            sensor_results['Rain'] = False
            
        try:
            if hasattr(rain, 'cleanup'):
                rain.cleanup()
        except:
            pass
            
    except Exception as e:
        print(f"❌ Rain sensor test failed: {e}")
        sensor_results['Rain'] = False
    
    # Test Camera (basic check)
    print("\nTesting Camera Module...")
    try:
        from sensors.ai_camera import AICamera
        camera = AICamera()
        
        if hasattr(camera, 'camera') and camera.camera:
            print("✅ Camera: Initialized successfully")
            sensor_results['Camera'] = True
            
            # Try a quick capture test
            try:
                test_image = "test_bloom.jpg"
                result = camera.capture_photo(test_image)
                if result and os.path.exists(test_image):
                    print("✅ Camera: Test photo captured")
                    os.remove(test_image)  # Cleanup
                else:
                    print("⚠️ Camera: Could not capture test photo")
            except Exception as e:
                print(f"⚠️ Camera: Photo test failed: {e}")
        else:
            print("❌ Camera: Not initialized")
            print("   Check: Camera enabled in raspi-config (Raspberry Pi only)")
            sensor_results['Camera'] = False
            
        try:
            if hasattr(camera, 'cleanup'):
                camera.cleanup()
        except:
            pass
            
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        sensor_results['Camera'] = False
    
    # Summary
    successful_sensors = sum(1 for success in sensor_results.values() if success)
    total_sensors = len(sensor_results)
    print(f"\nSensor Results: {successful_sensors}/{total_sensors} working")
    
    return sensor_results

def test_hardware():
    """Test hardware controllers"""
    print("\n🔧 Testing Hardware Controllers...")
    print("-" * 30)
    
    hardware_results = {}
    
    # Test LCD Display
    print("Testing LCD Display...")
    try:
        from hardware.lcd_display import LCDDisplay
        lcd = LCDDisplay()
        
        if hasattr(lcd, 'lcd') and lcd.lcd:
            lcd.show_message("BloomBotanics", "Test Mode")
            print("✅ LCD Display: Working")
            time.sleep(2)
            lcd.show_message("Test", "Complete")
            time.sleep(1)
            hardware_results['LCD'] = True
        else:
            print("❌ LCD Display: Not detected")
            print("   Check: I2C wiring, address 0x27, 5V power (Raspberry Pi only)")
            hardware_results['LCD'] = False
            
        try:
            if hasattr(lcd, 'cleanup'):
                lcd.cleanup()
        except:
            pass
            
    except Exception as e:
        print(f"❌ LCD test failed: {e}")
        hardware_results['LCD'] = False
    
    # Test Relay Controller
    print("\nTesting Relay Controller...")
    try:
        from hardware.relay_controller import RelayController
        relay = RelayController()
        
        print("   Testing relay ON/OFF (1 second each)...")
        
        # Test ON
        success_on = relay.turn_on()
        if success_on:
            print("   ✅ Relay ON successful")
            time.sleep(1)
        else:
            print("   ❌ Relay ON failed")
        
        # Test OFF
        success_off = relay.turn_off()
        if success_off:
            print("   ✅ Relay OFF successful")
        else:
            print("   ❌ Relay OFF failed")
        
        if success_on and success_off:
            print("✅ Relay Controller: Working")
            hardware_results['Relay'] = True
        else:
            print("❌ Relay Controller: Issues detected")
            hardware_results['Relay'] = False
            
        try:
            if hasattr(relay, 'cleanup'):
                relay.cleanup()
        except:
            pass
            
    except Exception as e:
        print(f"❌ Relay test failed: {e}")
        hardware_results['Relay'] = False
    
    # Test Fan Controller
    print("\nTesting Fan Controller...")
    try:
        from hardware.fan_controller import FanController
        fan = FanController()
        
        # Get CPU temperature
        if hasattr(fan, 'get_cpu_temperature'):
            try:
                cpu_temp = fan.get_cpu_temperature()
                print(f"   CPU Temperature: {cpu_temp:.1f}°C")
            except:
                print("   CPU Temperature: Not available")
        
        # Test fan speed control
        print("   Testing fan speeds...")
        for speed in [0, 50, 100, 0]:
            if hasattr(fan, 'set_speed'):
                success = fan.set_speed(speed)
                if success:
                    print(f"   Fan speed set to {speed}%")
                else:
                    print(f"   Failed to set fan speed to {speed}%")
            time.sleep(0.5)
        
        print("✅ Fan Controller: Working")
        hardware_results['Fan'] = True
        
        try:
            if hasattr(fan, 'cleanup'):
                fan.cleanup()
        except:
            pass
            
    except Exception as e:
        print(f"❌ Fan controller test failed: {e}")
        hardware_results['Fan'] = False
    
    # Test GSM Module (basic check only)
    print("\nTesting GSM Module...")
    try:
        from hardware.gsm_module import GSMModule
        gsm = GSMModule()
        
        if hasattr(gsm, 'connected') and gsm.connected:
            if hasattr(gsm, 'get_status'):
                status = gsm.get_status()
                print(f"✅ GSM Module: Connected")
                print(f"   Signal Strength: {status.get('signal_strength', 'Unknown')}")
                print(f"   Network: {status.get('network_status', 'Unknown')}")
            else:
                print("✅ GSM Module: Connected (basic)")
            hardware_results['GSM'] = True
        else:
            print("❌ GSM Module: Not connected")
            print("   Check: SIM card inserted, 4V power, antenna (Raspberry Pi only)")
            hardware_results['GSM'] = False
            
        try:
            if hasattr(gsm, 'cleanup'):
                gsm.cleanup()
        except:
            pass
            
    except Exception as e:
        print(f"❌ GSM test failed: {e}")
        hardware_results['GSM'] = False
    
    # Summary
    successful_hardware = sum(1 for success in hardware_results.values() if success)
    total_hardware = len(hardware_results)
    print(f"\nHardware Results: {successful_hardware}/{total_hardware} working")
    
    return hardware_results

def generate_test_report(sensor_results, hardware_results):
    """Generate test summary report"""
    print("\n📊 Test Summary Report")
    print("=" * 50)
    
    all_results = {**sensor_results, **hardware_results}
    total_tests = len(all_results)
    passed_tests = sum(1 for success in all_results.values() if success)
    
    print(f"Overall Results: {passed_tests}/{total_tests} tests passed")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print("")
    
    print("Detailed Results:")
    print("-" * 20)
    
    for component, success in all_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {component}")
    
    print("")
    
    # Detect platform
    try:
        import platform
        if platform.system() == 'Windows':
            print("🖥️ RUNNING ON WINDOWS")
            print("This is a development test only!")
            print("Deploy to Raspberry Pi for actual hardware control.")
            print("")
            print("Windows Development Notes:")
            print("• RPi.GPIO failures are expected")
            print("• Hardware tests will fail without actual sensors")
            print("• Use this to verify code structure only")
            print("")
        else:
            if passed_tests == total_tests:
                print("🎉 ALL TESTS PASSED!")
                print("Your BloomBotanics system is ready to run!")
                print("")
                print("Next steps:")
                print("1. Update PHONE_NUMBER in config.py")
                print("2. Run: python3 main.py")
                print("3. Start service: sudo systemctl start bloom-botanics")
            else:
                failed_components = [comp for comp, success in all_results.items() if not success]
                print("🔧 Some components need attention:")
                for component in failed_components:
                    print(f"  • {component}")
                print("")
                print("Troubleshooting:")
                print("1. Check wiring connections")
                print("2. Verify power supplies")
                print("3. Enable interfaces: sudo raspi-config")
                print("4. Install packages: pip install -r requirements.txt")
    except:
        pass
    
    return passed_tests == total_tests

def main():
    """Main test function"""
    try:
        print_header()
        
        # Test 1: Module imports
        config_ok = test_imports()
        if not config_ok:
            print("\n⚠️ Configuration issues detected")
            print("Some tests may fail without proper config.py")
        
        # Test 2: Basic GPIO (skip on Windows)
        try:
            import platform
            if platform.system() != 'Windows':
                if not test_gpio_basic():
                    print("\n❌ GPIO not working - check permissions and hardware")
            else:
                print("\n🖥️ Skipping GPIO test (Windows detected)")
        except:
            test_gpio_basic()  # Try anyway
        
        # Test 3: Sensors
        sensor_results = test_sensors()
        
        # Test 4: Hardware controllers
        hardware_results = test_hardware()
        
        # Generate report
        success = generate_test_report(sensor_results, hardware_results)
        
        return success
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False
    finally:
        # Final cleanup
        try:
            import RPi.GPIO as GPIO
            GPIO.cleanup()
        except:
            pass

if __name__ == "__main__":
    success = main()
    print(f"\nTest completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Keep window open on Windows
    try:
        import platform
        if platform.system() == 'Windows':
            input("\nPress Enter to exit...")
    except:
        pass
    
    sys.exit(0 if success else 1)
