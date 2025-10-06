"""PWM Fan Controller for Raspberry Pi Cooling"""

import RPi.GPIO as GPIO
import time
import threading
from utils.logger import get_logger
from config import FAN_CONTROL_PIN, CPU_TEMP_WARNING, CPU_TEMP_CRITICAL

class FanController:
    def __init__(self, fan_pin=FAN_CONTROL_PIN):
        self.logger = get_logger(__name__)
        self.fan_pin = fan_pin
        self.pwm = None
        self.current_speed = 0
        self.auto_control_enabled = True
        self.temp_history = []
        self.control_thread = None
        self.running = True
        
        self.setup_pwm()
        self.start_auto_control()
    
    def setup_pwm(self):
        """Setup PWM for fan speed control"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.fan_pin, GPIO.OUT)
            
            # Initialize PWM with 25kHz frequency (optimal for PC fans)
            self.pwm = GPIO.PWM(self.fan_pin, 25000)
            self.pwm.start(0)  # Start with fan OFF
            
            self.logger.info(f"Fan controller initialized on GPIO {self.fan_pin}")
            
        except Exception as e:
            self.logger.error(f"Fan PWM setup error: {e}")
    
    def set_speed(self, speed_percent):
        """Set fan speed (0-100%)"""
        if not self.pwm:
            return False
        
        try:
            # Clamp speed to valid range
            speed_percent = max(0, min(100, speed_percent))
            
            # Convert to duty cycle
            self.pwm.ChangeDutyCycle(speed_percent)
            self.current_speed = speed_percent
            
            if speed_percent != self.current_speed:
                self.logger.debug(f"Fan speed set to {speed_percent}%")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Fan speed control error: {e}")
            return False
    
    def get_cpu_temperature(self):
        """Get CPU temperature from thermal sensor"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_millicelsius = f.read().strip()
            
            temp_celsius = float(temp_millicelsius) / 1000.0
            
            # Add to history for averaging
            self.temp_history.append(temp_celsius)
            if len(self.temp_history) > 10:
                self.temp_history.pop(0)
            
            return temp_celsius
            
        except Exception as e:
            self.logger.error(f"CPU temperature reading error: {e}")
            return 50.0  # Default safe temperature
    
    def get_average_cpu_temperature(self):
        """Get average CPU temperature from recent readings"""
        if not self.temp_history:
            return self.get_cpu_temperature()
        
        return sum(self.temp_history) / len(self.temp_history)
    
    def calculate_fan_speed(self, cpu_temp):
        """Calculate optimal fan speed based on temperature"""
        try:
            # Temperature ranges and corresponding fan speeds
            if cpu_temp < 40:
                return 0  # No cooling needed
            elif cpu_temp < 50:
                return 25  # Low speed
            elif cpu_temp < 60:
                return 50  # Medium speed
            elif cpu_temp < 70:
                return 75  # High speed
            else:
                return 100  # Maximum cooling
                
        except Exception as e:
            self.logger.error(f"Fan speed calculation error: {e}")
            return 50  # Default medium speed
    
    def auto_control(self):
        """Single iteration of automatic fan control"""
        if not self.auto_control_enabled:
            return
        
        try:
            # Get current CPU temperature
            cpu_temp = self.get_cpu_temperature()
            
            # Calculate required fan speed
            required_speed = self.calculate_fan_speed(cpu_temp)
            
            # Apply hysteresis to prevent rapid speed changes
            if abs(required_speed - self.current_speed) > 5:
                self.set_speed(required_speed)
            
            # Log warning/critical temperatures
            if cpu_temp > CPU_TEMP_CRITICAL:
                self.logger.warning(f"🔥 CRITICAL CPU temperature: {cpu_temp:.1f}°C")
            elif cpu_temp > CPU_TEMP_WARNING:
                self.logger.info(f"🌡️ High CPU temperature: {cpu_temp:.1f}°C")
            
        except Exception as e:
            self.logger.error(f"Auto fan control error: {e}")
    
    def start_auto_control(self):
        """Start background thread for automatic fan control"""
        if self.control_thread and self.control_thread.is_alive():
            return
        
        def control_loop():
            while self.running:
                try:
                    self.auto_control()
                    time.sleep(10)  # Check every 10 seconds
                except Exception as e:
                    self.logger.error(f"Fan control loop error: {e}")
                    time.sleep(10)
        
        self.control_thread = threading.Thread(target=control_loop, daemon=True)
        self.control_thread.start()
        self.logger.info("Fan auto-control started")
    
    def stop_auto_control(self):
        """Stop automatic fan control"""
        self.auto_control_enabled = False
        self.logger.info("Fan auto-control stopped")
    
    def enable_auto_control(self):
        """Enable automatic fan control"""
        self.auto_control_enabled = True
        self.logger.info("Fan auto-control enabled")
    
    def test_fan(self):
        """Test fan operation"""
        try:
            self.logger.info("🧪 Testing fan operation...")
            
            # Disable auto control temporarily
            auto_was_enabled = self.auto_control_enabled
            self.auto_control_enabled = False
            
            # Test speed ramp
            test_speeds = [0, 25, 50, 75, 100, 50, 0]
            
            for speed in test_speeds:
                self.logger.info(f"Testing fan at {speed}%")
                self.set_speed(speed)
                time.sleep(2)
            
            # Restore auto control
            self.auto_control_enabled = auto_was_enabled
            
            self.logger.info("✅ Fan test completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Fan test failed: {e}")
            return False
    
    def get_status(self):
        """Get fan controller status"""
        try:
            cpu_temp = self.get_cpu_temperature()
            avg_temp = self.get_average_cpu_temperature()
            
            return {
                'current_speed': self.current_speed,
                'cpu_temperature': cpu_temp,
                'average_cpu_temperature': avg_temp,
                'auto_control_enabled': self.auto_control_enabled,
                'temp_history_size': len(self.temp_history),
                'status': 'running' if self.running else 'stopped'
            }
            
        except Exception as e:
            self.logger.error(f"Fan status error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_thermal_status(self):
        """Get detailed thermal status"""
        try:
            cpu_temp = self.get_cpu_temperature()
            
            if cpu_temp > CPU_TEMP_CRITICAL:
                status = "critical"
                recommendation = "System overheating! Check cooling."
            elif cpu_temp > CPU_TEMP_WARNING:
                status = "warning"
                recommendation = "Temperature elevated. Increase cooling."
            elif cpu_temp > 50:
                status = "normal"
                recommendation = "Temperature normal."
            else:
                status = "cool"
                recommendation = "System running cool."
            
            return {
                'temperature': cpu_temp,
                'status': status,
                'recommendation': recommendation,
                'fan_speed': self.current_speed
            }
            
        except Exception as e:
            self.logger.error(f"Thermal status error: {e}")
            return {'status': 'error'}
    
    def cleanup(self):
        """Cleanup fan controller resources"""
        try:
            self.logger.info("🔧 Cleaning up fan controller...")
            
            # Stop control thread
            self.running = False
            
            # Set fan to moderate speed before shutdown
            self.set_speed(30)
            time.sleep(1)
            
            # Stop PWM
            if self.pwm:
                self.pwm.stop()
            
            # Clean up GPIO
            GPIO.cleanup([self.fan_pin])
            
            self.logger.info("Fan controller cleaned up")
            
        except Exception as e:
            self.logger.error(f"Fan cleanup error: {e}")
