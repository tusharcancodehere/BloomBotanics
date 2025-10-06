"""SIM800L GSM Module for SMS Communication"""

import serial
import time
import threading
import queue
from utils.logger import get_logger
from config import GSM_SERIAL_PORT, GSM_BAUDRATE, GSM_TIMEOUT, PHONE_NUMBER

class GSMModule:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.serial = None
        self.connected = False
        self.signal_strength = 0
        self.network_status = "Unknown"
        self.sms_queue = queue.Queue()
        self.sms_thread = None
        
        self.initialize_gsm()
        self.start_sms_worker()
    
    def initialize_gsm(self):
        """Initialize GSM module with comprehensive setup"""
        try:
            # Open serial connection
            self.serial = serial.Serial(
                port=GSM_SERIAL_PORT,
                baudrate=GSM_BAUDRATE,
                timeout=GSM_TIMEOUT
            )
            
            self.logger.info(f"GSM serial opened on {GSM_SERIAL_PORT}")
            time.sleep(3)  # GSM module startup time
            
            # Initialize modem
            if self._initialize_modem():
                self.connected = True
                self.logger.info("✅ GSM module initialized successfully")
            else:
                self.logger.error("❌ GSM modem initialization failed")
                
        except Exception as e:
            self.logger.error(f"GSM initialization error: {e}")
            self.serial = None
    
    def _initialize_modem(self):
        """Initialize GSM modem with AT commands"""
        try:
            # Basic AT commands for initialization
            init_commands = [
                ("AT", "OK", "Basic communication"),
                ("ATE0", "OK", "Disable echo"),
                ("AT+CMGF=1", "OK", "Set SMS text mode"),
                ("AT+CNMI=2,2,0,0,0", "OK", "SMS notification setup"),
                ("AT+CSQ", "+CSQ:", "Signal quality check")
            ]
            
            for cmd, expected, description in init_commands:
                self.logger.info(f"GSM Init: {description}")
                if not self.send_at_command(cmd, expected):
                    self.logger.error(f"Failed: {description}")
                    return False
                time.sleep(1)
            
            # Get network status
            self._update_network_status()
            return True
            
        except Exception as e:
            self.logger.error(f"Modem initialization error: {e}")
            return False
    
    def send_at_command(self, command, expected="OK", timeout=5):
        """Send AT command with response validation"""
        if not self.serial:
            return False
        
        try:
            # Clear input buffer
            self.serial.reset_input_buffer()
            
            # Send command
            cmd_bytes = (command + '\r\n').encode('utf-8')
            self.serial.write(cmd_bytes)
            
            # Read response
            start_time = time.time()
            response = ""
            
            while time.time() - start_time < timeout:
                if self.serial.in_waiting > 0:
                    data = self.serial.read(self.serial.in_waiting)
                    response += data.decode('utf-8', errors='ignore')
                    
                    if expected in response:
                        self.logger.debug(f"AT Command Success: {command} -> {expected}")
                        return True
                        
                time.sleep(0.1)
            
            self.logger.warning(f"AT Command timeout: {command} (expected: {expected})")
            self.logger.debug(f"Response received: {response}")
            return False
            
        except Exception as e:
            self.logger.error(f"AT command error: {e}")
            return False
    
    def send_sms(self, message, phone_number=None, priority='normal'):
        """Queue SMS for sending"""
        if not phone_number:
            phone_number = PHONE_NUMBER
        
        sms_data = {
            'phone_number': phone_number,
            'message': message,
            'priority': priority,
            'timestamp': time.time(),
            'attempts': 0
        }
        
        # Add to queue based on priority
        if priority == 'high':
            # High priority messages go to front
            temp_queue = queue.Queue()
            temp_queue.put(sms_data)
            while not self.sms_queue.empty():
                temp_queue.put(self.sms_queue.get())
            self.sms_queue = temp_queue
        else:
            self.sms_queue.put(sms_data)
        
        self.logger.info(f"SMS queued for {phone_number}: {message[:50]}...")
        return True
    
    def _send_sms_direct(self, message, phone_number):
        """Send SMS directly via AT commands"""
        if not self.serial or not self.connected:
            return False
        
        try:
            # Set text mode
            if not self.send_at_command("AT+CMGF=1", "OK", 3):
                return False
            
            # Set recipient
            cmd = f'AT+CMGS="{phone_number}"'
            self.serial.write((cmd + '\r\n').encode())
            time.sleep(1)
            
            # Wait for '>' prompt
            if not self._wait_for_prompt('>', 10):
                self.logger.error("SMS: No prompt received")
                return False
            
            # Send message with Ctrl+Z
            message_bytes = (message + '\x1A').encode('utf-8')
            self.serial.write(message_bytes)
            
            # Wait for confirmation
            start_time = time.time()
            response = ""
            
            while time.time() - start_time < 30:  # SMS can take up to 30s
                if self.serial.in_waiting > 0:
                    data = self.serial.read(self.serial.in_waiting)
                    response += data.decode('utf-8', errors='ignore')
                    
                    if '+CMGS:' in response or 'OK' in response:
                        self.logger.info(f"📱 SMS sent successfully to {phone_number}")
                        return True
                    elif 'ERROR' in response:
                        self.logger.error(f"SMS sending failed: {response}")
                        return False
                        
                time.sleep(0.5)
            
            self.logger.error("SMS sending timeout")
            return False
            
        except Exception as e:
            self.logger.error(f"Direct SMS sending error: {e}")
            return False
    
    def start_sms_worker(self):
        """Start background thread for SMS sending"""
        if self.sms_thread and self.sms_thread.is_alive():
            return
        
        self.sms_thread = threading.Thread(target=self._sms_worker, daemon=True)
        self.sms_thread.start()
        self.logger.info("SMS worker thread started")
    
    def _sms_worker(self):
        """Background worker for SMS queue processing"""
        while True:
            try:
                # Get SMS from queue (blocking)
                sms_data = self.sms_queue.get(timeout=30)
                
                # Attempt to send SMS
                success = self._send_sms_direct(
                    sms_data['message'], 
                    sms_data['phone_number']
                )
                
                if not success:
                    sms_data['attempts'] += 1
                    
                    # Retry up to 3 times
                    if sms_data['attempts'] < 3:
                        self.logger.warning(f"SMS retry {sms_data['attempts']}/3")
                        time.sleep(5)  # Wait before retry
                        self.sms_queue.put(sms_data)
                    else:
                        self.logger.error("SMS failed after 3 attempts")
                
                self.sms_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"SMS worker error: {e}")
                time.sleep(5)
    
    def _wait_for_prompt(self, prompt, timeout=5):
        """Wait for specific prompt from modem"""
        start_time = time.time()
        response = ""
        
        while time.time() - start_time < timeout:
            if self.serial.in_waiting > 0:
                data = self.serial.read(self.serial.in_waiting)
                response += data.decode('utf-8', errors='ignore')
                if prompt in response:
                    return True
            time.sleep(0.1)
        
        return False
    
    def _update_network_status(self):
        """Update network registration and signal strength"""
        try:
            # Check signal strength
            if self.send_at_command("AT+CSQ", "+CSQ:"):
                time.sleep(1)
                response = self.serial.read(100).decode('utf-8', errors='ignore')
                if '+CSQ:' in response:
                    parts = response.split('+CSQ:')[1].split(',')
                    if parts:
                        self.signal_strength = int(parts[0].strip())
            
            # Check network registration
            if self.send_at_command("AT+CREG?", "+CREG:"):
                time.sleep(1)
                response = self.serial.read(100).decode('utf-8', errors='ignore')
                if '+CREG:' in response:
                    parts = response.split('+CREG:')[1].split(',')
                    if len(parts) > 1:
                        reg_status = int(parts[1].strip())
                        if reg_status == 1:
                            self.network_status = "Registered (home)"
                        elif reg_status == 5:
                            self.network_status = "Registered (roaming)"
                        else:
                            self.network_status = "Not registered"
            
        except Exception as e:
            self.logger.error(f"Network status update error: {e}")
    
    def get_status(self):
        """Get comprehensive GSM status"""
        self._update_network_status()
        
        return {
            'connected': self.connected,
            'signal_strength': self.signal_strength,
            'network_status': self.network_status,
            'sms_queue_size': self.sms_queue.qsize(),
            'serial_port': GSM_SERIAL_PORT,
            'baudrate': GSM_BAUDRATE
        }
    
    def test_gsm(self):
        """Test GSM functionality"""
        try:
            self.logger.info("🧪 Testing GSM module...")
            
            # Test basic AT command
            if not self.send_at_command("AT", "OK"):
                self.logger.error("❌ Basic AT command failed")
                return False
            
            # Test signal strength
            if not self.send_at_command("AT+CSQ", "+CSQ:"):
                self.logger.error("❌ Signal strength check failed")
                return False
            
            # Test network registration
            self._update_network_status()
            
            status = self.get_status()
            self.logger.info(f"✅ GSM test completed: {status}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ GSM test failed: {e}")
            return False
    
    def cleanup(self):
        """Cleanup GSM resources"""
        try:
            self.logger.info("🔧 Cleaning up GSM module...")
            
            # Stop SMS worker
            if self.sms_thread and self.sms_thread.is_alive():
                # Note: Daemon thread will stop when main program exits
                pass
            
            # Close serial connection
            if self.serial:
                try:
                    self.serial.close()
                    self.logger.info("GSM serial connection closed")
                except:
                    pass
            
            self.connected = False
            
        except Exception as e:
            self.logger.error(f"GSM cleanup error: {e}")
