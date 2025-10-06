"""Relay Controller for Irrigation and Actuators"""

import RPi.GPIO as GPIO
import time
import threading
from utils.logger import get_logger
from config import RELAY_PIN, IRRIGATION_DURATION

class RelayController:
    def __init__(self, relay_pin=RELAY_PIN):
        self.logger = get_logger(__name__)
        self.relay_pin = relay_pin
        self.is_active = False
        self.activation_count = 0
        self.total_runtime = 0
        self.last_activation = 0
        self.active_timer = None
        
        self.setup_gpio()
    
    def setup_gpio(self):
        """Setup GPIO for relay control"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.relay_pin, GPIO.OUT)
            GPIO.output(self.relay_pin, GPIO.LOW)  # Start with relay OFF (NC)
            
            self.logger.info(f"Relay controller initialized on GPIO {self.relay_pin}")
            
        except Exception as e:
            self.logger.error(f"Relay GPIO setup error: {e}")
    
    def turn_on(self):
        """Turn relay ON (activate irrigation)"""
        try:
            if not self.is_active:
                GPIO.output(self.relay_pin, GPIO.HIGH)
                self.is_active = True
                self.last_activation = time.time()
                self.activation_count += 1
                
                self.logger.info("🔛 Relay activated - Irrigation ON")
                return True
            else:
                self.logger.warning("Relay already active")
                return False
                
        except Exception as e:
            self.logger.error(f"Relay activation error: {e}")
            return False
    
    def turn_off(self):
        """Turn relay OFF (stop irrigation)"""
        try:
            if self.is_active:
                GPIO.output(self.relay_pin, GPIO.LOW)
                
                # Calculate runtime
                runtime = time.time() - self.last_activation
                self.total_runtime += runtime
                self.is_active = False
                
                self.logger.info(f"⏹️ Relay deactivated - Runtime: {runtime:.1f}s")
                return True
            else:
                self.logger.warning("Relay already inactive")
                return False
                
        except Exception as e:
            self.logger.error(f"Relay deactivation error: {e}")
            return False
    
    def activate(self, duration=IRRIGATION_DURATION):
        """Activate relay for specified duration"""
        if self.is_active:
            self.logger.warning("Irrigation already running")
            return False
        
        try:
            self.turn_on()
            
            # Create timer to turn off relay
            self.active_timer = threading.Timer(duration, self.turn_off)
            self.active_timer.daemon = True
            self.active_timer.start()
            
            self.logger.info(f"💧 Irrigation started for {duration} seconds")
            return True
            
        except Exception as e:
            self.logger.error(f"Timed activation error: {e}")
            return False
    
    def auto_irrigation(self, duration=IRRIGATION_DURATION):
        """Automatic irrigation with enhanced logging"""
        if self.is_active:
            self.logger.warning("Auto-irrigation blocked - already running")
            return False
        
        try:
            self.logger.info(f"🤖 Auto-irrigation triggered - Duration: {duration}s")
            success = self.activate(duration)
            
            if success:
                # Log irrigation event
                irrigation_data = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': duration,
                    'trigger': 'automatic',
                    'activation_count': self.activation_count
                }
                
                # Save irrigation log (could be enhanced to save to file)
                self.logger.info(f"💧 Irrigation log: {irrigation_data}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Auto-irrigation error: {e}")
            return False
    
    def manual_irrigation(self, duration=30):
        """Manual irrigation trigger"""
        try:
            self.logger.info(f"👤 Manual irrigation triggered - Duration: {duration}s")
            success = self.activate(duration)
            
            if success:
                irrigation_data = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': duration,
                    'trigger': 'manual',
                    'activation_count': self.activation_count
                }
                self.logger.info(f"💧 Manual irrigation log: {irrigation_data}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Manual irrigation error: {e}")
            return False
    
    def emergency_stop(self):
        """Emergency stop - immediately turn off relay"""
        try:
            if self.active_timer:
                self.active_timer.cancel()
                self.active_timer = None
            
            success = self.turn_off()
            if success:
                self.logger.warning("🚨 Emergency stop - Irrigation halted")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Emergency stop error: {e}")
            return False
    
    def get_status(self):
        """Get relay status and statistics"""
        try:
            current_runtime = 0
            if self.is_active:
                current_runtime = time.time() - self.last_activation
            
            return {
                'is_active': self.is_active,
                'activation_count': self.activation_count,
                'total_runtime': self.total_runtime,
                'current_runtime': current_runtime,
                'last_activation': self.last_activation,
                'status': 'active' if self.is_active else 'inactive'
            }
            
        except Exception as e:
            self.logger.error(f"Status check error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def test_relay(self, test_duration=2):
        """Test relay operation"""
        try:
            self.logger.info("🧪 Testing relay operation...")
            
            # Test ON
            self.turn_on()
            time.sleep(test_duration)
            
            # Test OFF
            self.turn_off()
            
            self.logger.info("✅ Relay test completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Relay test failed: {e}")
            return False
    
    def get_daily_stats(self):
        """Get daily irrigation statistics"""
        # This would typically read from a log file
        # For now, return current session stats
        return {
            'daily_activations': self.activation_count,
            'daily_runtime': self.total_runtime,
            'average_duration': self.total_runtime / max(1, self.activation_count),
            'last_irrigation': time.strftime('%H:%M:%S', time.localtime(self.last_activation)) if self.last_activation else None
        }
    
    def cleanup(self):
        """Cleanup relay resources"""
        try:
            # Cancel active timer
            if self.active_timer:
                self.active_timer.cancel()
            
            # Turn off relay
            self.turn_off()
            
            # Clean up GPIO
            GPIO.cleanup([self.relay_pin])
            
            # Log final statistics
            stats = self.get_status()
            self.logger.info(f"🔧 Relay cleanup - Final stats: {stats}")
            
        except Exception as e:
            self.logger.error(f"Relay cleanup error: {e}")
