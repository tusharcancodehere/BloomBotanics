#!/usr/bin/env python3
"""
🌱 BloomBotanics Agricultural Monitoring System
Complete Production Code - Version 2.0

Features:
✅ Real-time sensor monitoring (DHT22, Soil, Rain)
✅ AI-powered animal/human detection  
✅ Automatic irrigation control
✅ SMS alerts for threats and system status
✅ LCD display for local monitoring
✅ Cooling fan control
✅ Data logging and image capture
✅ Pi Connect remote access
✅ Auto-start on boot
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
    from sensors.ai_camera import AICamera
    from hardware.lcd_display import LCDDisplay
    from hardware.relay_controller import RelayController
    from hardware.gsm_module import GSMModule
    from hardware.fan_controller import FanController
    from utils.logger import get_logger
    from utils.helpers import SystemHealth, DataManager
    from config import *
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure all required packages are installed")
    sys.exit(1)

class BloomBotanicsSystem:
    """
    🌱 Main BloomBotanics Agricultural Monitoring System
    
    Complete autonomous agricultural monitoring with:
    - Sensor monitoring and data logging
    - AI threat detection and alerts  
    - Automatic irrigation control
    - Remote access via Pi Connect
    - SMS notifications and status updates
    - System health monitoring and recovery
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
        self.rain_sensor = None
        self.ai_camera = None
        self.lcd = None
        self.relay = None
        self.gsm = None
        self.fan = None
        
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
        
        initialization_status = {}
        
        # Initialize sensors with individual error handling
        try:
            self.dht22 = DHT22Sensor()
            initialization_status['DHT22'] = 'OK' if self.dht22 else 'FAIL'
        except Exception as e:
            self.logger.error(f"DHT22 initialization failed: {e}")
            initialization_status['DHT22'] = 'FAIL'
            
        try:
            self.soil_sensor = SoilMoistureSensor()
            initialization_status['Soil'] = 'OK' if self.soil_sensor else 'FAIL'
        except Exception as e:
            self.logger.error(f"Soil sensor initialization failed: {e}")
            initialization_status['Soil'] = 'FAIL'
            
        try:
            self.rain_sensor = RainSensor()
            initialization_status['Rain'] = 'OK' if self.rain_sensor else 'FAIL'
        except Exception as e:
            self.logger.error(f"Rain sensor initialization failed: {e}")
            initialization_status['Rain'] = 'FAIL'
            
        try:
            if AI_DETECTION_ENABLED:
                self.ai_camera = AICamera()
                initialization_status['AI_Camera'] = 'OK' if self.ai_camera else 'FAIL'
            else:
                initialization_status['AI_Camera'] = 'DISABLED'
        except Exception as e:
            self.logger.error(f"AI Camera initialization failed: {e}")
            initialization_status['AI_Camera'] = 'FAIL'
            
        # Initialize hardware controllers
        try:
            self.relay = RelayController()
            initialization_status['Relay'] = 'OK' if self.relay else 'FAIL'
        except Exception as e:
            self.logger.error(f"Relay initialization failed: {e}")
            initialization_status['Relay'] = 'FAIL'
            
        try:
            self.gsm = GSMModule()
            initialization_status['GSM'] = 'OK' if self.gsm else 'FAIL'
        except Exception as e:
            self.logger.error(f"GSM initialization failed: {e}")
            initialization_status['GSM'] = 'FAIL'
            
        try:
            self.fan = FanController()
            initialization_status['Fan'] = 'OK' if self.fan else 'FAIL'
        except Exception as e:
            self.logger.error(f"Fan controller initialization failed: {e}")
            initialization_status['Fan'] = 'FAIL'
        
        # Log initialization results
        self.logger.info("🔍 Component Initialization Status:")
        for component, status in initialization_status.items():
            status_emoji = "✅" if status == "OK" else "❌" if status == "FAIL" else "⚠️"
            self.logger.info(f"  {status_emoji} {component}: {status}")
        
        # Check if critical components failed
        critical_components = ['DHT22', 'Soil', 'Relay', 'GSM']
        failed_critical = [comp for comp in critical_components if initialization_status.get(comp) == 'FAIL']
        
        if failed_critical:
            error_msg = f"Critical components failed: {', '.join(failed_critical)}"
            self.logger.error(f"❌ {error_msg}")
            if self.lcd:
                self.lcd.show_message("INIT ERROR!", f"Failed: {len(failed_critical)}")
            
            # Send error SMS if GSM is working
            if self.gsm and initialization_status.get('GSM') == 'OK':
                self.gsm.send_sms(SMS_TEMPLATES['system_error'].format(
                    error_type="Initialization Failure",
                    error_msg=error_msg,
                    timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ))
            
            # Don't exit - try to run with available components
            
        # Send startup notification
        if self.gsm:
            startup_msg = SMS_TEMPLATES['startup'].format(
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            self.gsm.send_sms(startup_msg)
        
        if self.lcd:
            self.lcd.show_message("BloomBotanics", "System Ready!")
            time.sleep(2)
        
        self.logger.info("🌱 BloomBotanics System Initialization Complete!")
        return initialization_status
    
    def read_all_sensors(self):
        """Read data from all sensors with error handling"""
        sensor_data = {
            'timestamp': datetime.now().isoformat(),
            'temperature': None,
            'humidity': None,
            'soil_moisture': None,
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
        
        # Read Soil Moisture
        if self.soil_sensor:
            try:
                soil_data = self.soil_sensor.read_moisture()
                if soil_data and soil_data.get('status') == 'success':
                    sensor_data['soil_moisture'] = soil_data['moisture_level']
                else:
                    sensor_data['errors'].append('Soil sensor reading failed')
            except Exception as e:
                sensor_data['errors'].append(f'Soil sensor error: {str(e)}')
                self.logger.error(f"Soil sensor reading error: {e}")
        
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
    
    def check_sensor_alerts(self, sensor_data):
        """Check sensor thresholds and generate alerts"""
        alerts = []
        current_time = time.time()
        
        # Temperature alerts
        if sensor_data['temperature'] is not None:
            temp = sensor_data['temperature']
            if temp < TEMP_MIN:
                alerts.append({
                    'type': 'temperature_low',
                    'message': f"🌡️ Low temperature: {temp}°C (Min: {TEMP_MIN}°C)",
                    'value': temp,
                    'threshold': TEMP_MIN,
                    'severity': 'warning'
                })
            elif temp > TEMP_MAX:
                alerts.append({
                    'type': 'temperature_high', 
                    'message': f"🌡️ High temperature: {temp}°C (Max: {TEMP_MAX}°C)",
                    'value': temp,
                    'threshold': TEMP_MAX,
                    'severity': 'warning'
                })
        
        # Humidity alerts  
        if sensor_data['humidity'] is not None:
            humidity = sensor_data['humidity']
            if humidity < HUMIDITY_MIN:
                alerts.append({
                    'type': 'humidity_low',
                    'message': f"💧 Low humidity: {humidity}% (Min: {HUMIDITY_MIN}%)",
                    'value': humidity,
                    'threshold': HUMIDITY_MIN,
                    'severity': 'warning'
                })
            elif humidity > HUMIDITY_MAX:
                alerts.append({
                    'type': 'humidity_high',
                    'message': f"💧 High humidity: {humidity}% (Max: {HUMIDITY_MAX}%)",
                    'value': humidity,
                    'threshold': HUMIDITY_MAX,
                    'severity': 'warning'
                })
        
        # Soil moisture - trigger irrigation
        if sensor_data['soil_moisture'] is not None:
            soil_moisture = sensor_data['soil_moisture']
            if (soil_moisture < SOIL_MOISTURE_MIN and 
                not sensor_data['rain_detected'] and
                AUTO_IRRIGATION and
                current_time - self.last_irrigation > IRRIGATION_COOLDOWN):
                
                # Trigger irrigation
                try:
                    if self.relay:
                        self.relay.auto_irrigation(IRRIGATION_DURATION)
                        self.last_irrigation = current_time
                        
                        irrigation_msg = SMS_TEMPLATES['irrigation'].format(
                            moisture=soil_moisture,
                            duration=IRRIGATION_DURATION,
                            timestamp=datetime.now().strftime('%H:%M:%S')
                        )
                        
                        alerts.append({
                            'type': 'irrigation_triggered',
                            'message': irrigation_msg,
                            'value': soil_moisture,
                            'threshold': SOIL_MOISTURE_MIN,
                            'severity': 'info'
                        })
                        
                        if self.gsm:
                            self.gsm.send_sms(irrigation_msg)
                        
                        self.logger.info(f"💧 Auto-irrigation triggered - Soil: {soil_moisture}%")
                        
                except Exception as e:
                    self.logger.error(f"Irrigation trigger error: {e}")
            
            elif soil_moisture < SOIL_MOISTURE_MIN:
                # Low moisture but can't irrigate (rain or cooldown)
                reason = "rain detected" if sensor_data['rain_detected'] else "irrigation cooldown active"
                alerts.append({
                    'type': 'soil_moisture_low',
                    'message': f"🌱 Low soil moisture: {soil_moisture}% ({reason})",
                    'value': soil_moisture,
                    'threshold': SOIL_MOISTURE_MIN,
                    'severity': 'warning'
                })
        
        # Rain detection alert
        if sensor_data['rain_detected']:
            alerts.append({
                'type': 'rain_detected',
                'message': "☔ Rain detected - irrigation paused",
                'value': True,
                'threshold': None,
                'severity': 'info'
            })
        
        # CPU temperature warning
        if sensor_data['cpu_temperature'] and sensor_data['cpu_temperature'] > CPU_TEMP_WARNING:
            alerts.append({
                'type': 'cpu_temperature_high',
                'message': f"🔥 High CPU temperature: {sensor_data['cpu_temperature']:.1f}°C",
                'value': sensor_data['cpu_temperature'],
                'threshold': CPU_TEMP_WARNING,
                'severity': 'critical' if sensor_data['cpu_temperature'] > CPU_TEMP_CRITICAL else 'warning'
            })
        
        # System errors
        if sensor_data['errors']:
            alerts.append({
                'type': 'system_errors',
                'message': f"⚠️ System errors: {len(sensor_data['errors'])} components affected",
                'value': len(sensor_data['errors']),
                'threshold': 0,
                'severity': 'critical' if len(sensor_data['errors']) > 2 else 'warning'
            })
        
        # Send SMS alerts for critical issues (with cooldown)
        critical_alerts = [alert for alert in alerts if alert['severity'] in ['critical', 'warning']]
        if critical_alerts and current_time - self.last_sensor_alert > ALERT_COOLDOWN:
            
            # Prepare SMS message
            alert_messages = [alert['message'] for alert in critical_alerts[:3]]  # Limit to 3 alerts
            sms_text = f"🚨 {FARM_NAME} Alert:\n" + "\n".join(alert_messages)
            
            if self.gsm:
                try:
                    self.gsm.send_sms(sms_text)
                    self.last_sensor_alert = current_time
                    self.logger.info("📱 Sensor alert SMS sent")
                except Exception as e:
                    self.logger.error(f"Failed to send sensor alert SMS: {e}")
        
        return alerts
    
    def process_ai_detections(self):
        """Process AI detection results with enhanced threat response"""
        if not AI_DETECTION_ENABLED or not self.ai_camera:
            return []
        
        try:
            detections, frame = self.ai_camera.detect_threats()
            
            if not detections:
                return []
            
            self.logger.info(f"🤖 AI Detection: {len(detections)} objects detected")
            
            # Save detection image
            detection_image = None
            try:
                detection_image = self.ai_camera.save_detection_image(frame, detections)
            except Exception as e:
                self.logger.error(f"Failed to save detection image: {e}")
            
            # Process each detection
            critical_detections = []
            for detection in detections:
                class_id = detection['class_id']
                threat_level = detection['threat_level']
                
                if threat_level == 'critical' and class_id in CRITICAL_THREATS:
                    critical_detections.append(detection)
                    
                    # Show alert on LCD
                    if self.lcd:
                        threat_name = CRITICAL_THREATS[class_id]['name'].upper()
                        self.lcd.show_message("🚨 THREAT!", threat_name)
                        time.sleep(3)
                    
                    # Log threat
                    threat_info = CRITICAL_THREATS[class_id]
                    self.logger.warning(f"🚨 Critical threat detected: {threat_info['message']}")
            
            # Send SMS for critical threats
            if critical_detections:
                self._send_threat_alerts(critical_detections, detection_image)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"❌ AI detection error: {e}")
            return []
    
    def _send_threat_alerts(self, detections, image_path):
        """Send SMS alerts for detected threats"""
        current_time = time.time()
        
        # Limit threat alerts frequency (10 minutes cooldown)
        if current_time - self.last_detection_alert < 600:
            return
        
        if not self.gsm:
            return
        
        try:
            # Prepare threat message
            threat_types = []
            for detection in detections:
                class_id = detection['class_id']
                if class_id in CRITICAL_THREATS:
                    threat_types.append(CRITICAL_THREATS[class_id]['message'])
            
            # Remove duplicates and limit
            unique_threats = list(set(threat_types))[:2]
            
            sms_message = SMS_TEMPLATES['threat_critical'].format(
                threat_type="\n".join(unique_threats),
                farm_name=FARM_NAME,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
            # Add image info if available
            if image_path:
                image_name = os.path.basename(image_path)
                sms_message += f"\nEvidence: {image_name}"
            
            # Send SMS
            if self.gsm.send_sms(sms_message):
                self.last_detection_alert = current_time
                self.logger.info(f"🚨 Threat alert SMS sent: {len(detections)} threats")
            else:
                self.logger.error("Failed to send threat alert SMS")
                
        except Exception as e:
            self.logger.error(f"❌ Threat alert SMS error: {e}")
    
    def capture_periodic_photo(self):
        """Capture regular crop monitoring photos"""
        current_time = time.time()
        
        if current_time - self.last_photo > PHOTO_INTERVAL and self.ai_camera:
            try:
                photo_path = self.ai_camera.capture_photo()
                if photo_path:
                    self.last_photo = current_time
                    self.logger.info(f"📸 Periodic photo: {os.path.basename(photo_path)}")
                    return photo_path
            except Exception as e:
                self.logger.error(f"Photo capture error: {e}")
        
        return None
    
    def update_lcd_display(self, sensor_data, alerts=None):
        """Update LCD with current status and alerts"""
        if not self.lcd:
            return
        
        try:
            # Show critical alerts first
            if alerts:
                critical_alerts = [a for a in alerts if a['severity'] == 'critical']
                if critical_alerts:
                    alert = critical_alerts[0]
                    self.lcd.show_message("🚨 CRITICAL!", alert['type'].replace('_', ' ').title())
                    time.sleep(3)
                    return
            
            # Show normal sensor data
            temp = sensor_data.get('temperature', 0) or 0
            humidity = sensor_data.get('humidity', 0) or 0
            soil = sensor_data.get('soil_moisture', 0) or 0
            rain = sensor_data.get('rain_detected', False)
            
            self.lcd.display_sensor_data(temp, humidity, soil, rain)
            
        except Exception as e:
            self.logger.error(f"LCD update error: {e}")
    
    def check_system_health(self):
        """Monitor system health and performance"""
        current_time = time.time()
        
        if current_time - self.last_health_check < HEALTH_CHECK_INTERVAL:
            return
        
        try:
            health_status = self.health_monitor.get_system_health()
            
            # Log health status
            self.logger.info(f"💓 System Health Check - CPU: {health_status['cpu_temp']:.1f}°C, "
                           f"Memory: {health_status['memory_usage']:.1f}%, "
                           f"Disk: {health_status['disk_usage']:.1f}%")
            
            # Check for health issues
            issues = []
            if health_status['cpu_temp'] > CPU_TEMP_CRITICAL:
                issues.append(f"CPU temperature critical: {health_status['cpu_temp']:.1f}°C")
            
            if health_status['memory_usage'] > MEMORY_WARNING:
                issues.append(f"High memory usage: {health_status['memory_usage']:.1f}%")
            
            if health_status['disk_usage'] > DISK_WARNING:
                issues.append(f"Low disk space: {health_status['disk_usage']:.1f}%")
            
            # Send health alerts if needed
            if issues and self.gsm:
                health_alert = f"🏥 {FARM_NAME} Health Alert:\n" + "\n".join(issues[:2])
                self.gsm.send_sms(health_alert)
            
            self.last_health_check = current_time
            
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
    
    def send_daily_report(self):
        """Send daily system report"""
        if not DAILY_REPORT_ENABLED or not self.gsm:
            return
        
        current_date = datetime.now().date()
        
        # Check if daily report already sent today
        if self.last_daily_report == current_date:
            return
        
        # Check if it's time for daily report
        current_time = datetime.now().time()
        report_time = datetime.strptime(DAILY_REPORT_TIME, "%H:%M").time()
        
        if current_time < report_time:
            return
        
        try:
            # Get daily statistics
            daily_stats = self.data_manager.get_daily_statistics()
            
            report_message = SMS_TEMPLATES['daily_report'].format(
                avg_temp=daily_stats.get('avg_temperature', 0),
                avg_humidity=daily_stats.get('avg_humidity', 0),
                irrigation_count=daily_stats.get('irrigation_count', 0),
                threat_count=daily_stats.get('threat_count', 0)
            )
            
            if self.gsm.send_sms(report_message):
                self.last_daily_report = current_date
                self.logger.info("📊 Daily report sent")
            
        except Exception as e:
            self.logger.error(f"Daily report error: {e}")
    
    def main_monitoring_loop(self):
        """Main system monitoring loop with comprehensive error handling"""
        self.logger.info("🔄 Starting main monitoring loop...")
        self.running = True
        
        try:
            while self.running:
                loop_start_time = time.time()
                self.loop_count += 1
                
                try:
                    # Read all sensors
                    sensor_data = self.read_all_sensors()
                    
                    # Save sensor data
                    self.data_manager.save_sensor_data(sensor_data)
                    
                    # Check sensor alerts and irrigation
                    sensor_alerts = self.check_sensor_alerts(sensor_data)
                    
                    # AI threat detection
                    ai_detections = []
                    if self.loop_count % 2 == 0:  # Every other loop for performance
                        ai_detections = self.process_ai_detections()
                    
                    # Update LCD display
                    self.update_lcd_display(sensor_data, sensor_alerts)
                    
                    # Capture periodic photos
                    if self.loop_count % 60 == 0:  # Every hour
                        self.capture_periodic_photo()
                    
                    # System health monitoring
                    self.check_system_health()
                    
                    # Control cooling fan
                    if self.fan:
                        self.fan.auto_control()
                    
                    # Send daily report
                    self.send_daily_report()
                    
                    # Log status periodically
                    if self.loop_count % 10 == 0:  # Every 10 minutes
                        runtime = (time.time() - self.start_time) / 3600  # Hours
                        self.logger.info(f"📈 System Status - Loop: {self.loop_count}, "
                                       f"Runtime: {runtime:.1f}h, Errors: {self.error_count}")
                    
                except Exception as e:
                    self.error_count += 1
                    self.logger.error(f"❌ Main loop error #{self.error_count}: {e}")
                    self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
                    
                    # Show error on LCD
                    if self.lcd:
                        self.lcd.show_message("SYSTEM ERROR", f"Error #{self.error_count}")
                        time.sleep(2)
                    
                    # If too many errors, attempt restart
                    if self.error_count > 10 and AUTO_RESTART_ON_ERROR:
                        self.logger.critical("🔄 Too many errors - attempting system restart")
                        if self.gsm:
                            self.gsm.send_sms(f"🔄 {FARM_NAME} System restarting due to errors")
                        self._restart_system()
                        break
                
                # Calculate sleep time to maintain consistent intervals
                loop_duration = time.time() - loop_start_time
                sleep_time = max(0, SENSOR_READ_INTERVAL - loop_duration)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    self.logger.warning(f"⚠️ Loop took {loop_duration:.1f}s, longer than interval {SENSOR_READ_INTERVAL}s")
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Manual shutdown requested")
        except Exception as e:
            self.logger.critical(f"💥 Fatal system error: {e}")
            self.logger.critical(f"💥 Fatal traceback: {traceback.format_exc()}")
            
            # Send critical error SMS
            if self.gsm:
                self.gsm.send_sms(SMS_TEMPLATES['system_error'].format(
                    error_type="Fatal System Error",
                    error_msg=str(e)[:100],
                    timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ))
        finally:
            self.cleanup()
    
    def _restart_system(self):
        """Restart the system components"""
        if self.restart_count >= MAX_RESTART_ATTEMPTS:
            self.logger.critical("🚨 Maximum restart attempts reached - stopping system")
            return
        
        self.restart_count += 1
        self.logger.info(f"🔄 System restart attempt {self.restart_count}/{MAX_RESTART_ATTEMPTS}")
        
        # Cleanup current components
        self.cleanup(send_notification=False)
        
        # Wait before restart
        time.sleep(RESTART_COOLDOWN)
        
        # Reinitialize system
        self.error_count = 0
        self.initialize_system()
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"🛑 Received signal {signum} - initiating graceful shutdown...")
        self.running = False
    
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
                ('Soil Sensor', self.soil_sensor),
                ('Rain Sensor', self.rain_sensor),
                ('AI Camera', self.ai_camera),
                ('LCD Display', self.lcd),
                ('Relay', self.relay),
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
