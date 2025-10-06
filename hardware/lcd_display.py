"""16x2 I2C LCD Display Controller"""

import time
import threading
from RPLCD.i2c import CharLCD
from utils.logger import get_logger
from config import LCD_I2C_ADDRESS, LCD_COLUMNS, LCD_ROWS

class LCDDisplay:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.lcd = None
        self.display_lock = threading.Lock()
        self.current_message = None
        self.scroll_position = 0
        self.last_update = 0
        
        self.initialize_lcd()
    
    def initialize_lcd(self):
        """Initialize I2C LCD with error handling"""
        try:
            # Try to initialize LCD
            self.lcd = CharLCD(
                i2c_expander='PCF8574',
                address=LCD_I2C_ADDRESS,
                port=1,
                cols=LCD_COLUMNS,
                rows=LCD_ROWS,
                backlight_enabled=True
            )
            
            # Test LCD
            self.lcd.clear()
            self.show_message("BloomBotanics", "Initializing...")
            time.sleep(1)
            
            self.logger.info(f"LCD initialized on I2C address {hex(LCD_I2C_ADDRESS)}")
            
        except Exception as e:
            self.logger.error(f"LCD initialization failed: {e}")
            self.logger.warning("System will continue without LCD display")
            self.lcd = None
    
    def show_message(self, line1, line2="", scroll=False):
        """Display message on LCD with optional scrolling"""
        if not self.lcd:
            return
        
        try:
            with self.display_lock:
                self.lcd.clear()
                
                # Handle scrolling for long text
                if scroll and len(line1) > LCD_COLUMNS:
                    line1 = self._scroll_text(line1)
                if scroll and len(line2) > LCD_COLUMNS:
                    line2 = self._scroll_text(line2)
                
                # Display first line
                self.lcd.write_string(line1[:LCD_COLUMNS])
                
                # Display second line if provided
                if line2:
                    self.lcd.crlf()
                    self.lcd.write_string(line2[:LCD_COLUMNS])
                    
                self.current_message = (line1, line2)
                self.last_update = time.time()
                
        except Exception as e:
            self.logger.error(f"LCD display error: {e}")
    
    def display_sensor_data(self, temp, humidity, soil_moisture, rain_detected):
        """Display sensor data in optimized format"""
        if not self.lcd:
            return
        
        try:
            # Format sensor data for 16x2 display
            # Line 1: Temperature and Humidity
            line1 = f"T:{temp:4.1f}C H:{humidity:2.0f}%"
            
            # Line 2: Soil moisture and Rain
            rain_symbol = "R:Y" if rain_detected else "R:N"
            line2 = f"S:{soil_moisture:2.0f}% {rain_symbol}"
            
            self.show_message(line1, line2)
            
        except Exception as e:
            self.logger.error(f"Sensor data display error: {e}")
    
    def show_threat_alert(self, threat_type, duration=3):
        """Display threat alert with flashing effect"""
        if not self.lcd:
            return
        
        try:
            original_message = self.current_message
            
            # Flash alert message
            for _ in range(duration):
                self.show_message("🚨 THREAT!", threat_type.upper())
                time.sleep(0.5)
                self.lcd.clear()
                time.sleep(0.5)
            
            # Restore original message
            if original_message:
                self.show_message(original_message[0], original_message[1])
                
        except Exception as e:
            self.logger.error(f"Threat alert display error: {e}")
    
    def show_system_status(self, status_type):
        """Display system status messages"""
        status_messages = {
            'starting': ("BloomBotanics", "Starting..."),
            'running': ("System", "Running OK"),
            'error': ("System Error", "Check Logs"),
            'irrigation': ("Irrigation", "Active"),
            'offline': ("System", "Offline"),
            'updating': ("System", "Updating..."),
            'rebooting': ("System", "Rebooting...")
        }
        
        message = status_messages.get(status_type, ("Unknown", "Status"))
        self.show_message(message[0], message[1])
    
    def display_time_and_status(self):
        """Display current time and system status"""
        if not self.lcd:
            return
        
        try:
            current_time = time.strftime('%H:%M:%S')
            current_date = time.strftime('%m/%d')
            
            line1 = f"Time: {current_time}"
            line2 = f"Date: {current_date}"
            
            self.show_message(line1, line2)
            
        except Exception as e:
            self.logger.error(f"Time display error: {e}")
    
    def _scroll_text(self, text):
        """Handle text scrolling for long messages"""
        if len(text) <= LCD_COLUMNS:
            return text
        
        # Simple scrolling implementation
        scroll_text = text + "    "  # Add padding
        start_pos = self.scroll_position % len(scroll_text)
        scrolled = scroll_text[start_pos:start_pos + LCD_COLUMNS]
        
        if len(scrolled) < LCD_COLUMNS:
            scrolled += scroll_text[:LCD_COLUMNS - len(scrolled)]
        
        self.scroll_position += 1
        return scrolled
    
    def set_backlight(self, enabled=True):
        """Control LCD backlight"""
        if self.lcd:
            try:
                self.lcd.backlight_enabled = enabled
            except Exception as e:
                self.logger.error(f"Backlight control error: {e}")
    
    def create_custom_characters(self):
        """Create custom characters for agricultural symbols"""
        if not self.lcd:
            return
        
        try:
            # Define custom characters (8x5 pixel patterns)
            # Temperature symbol
            temp_char = [
                0b00100,
                0b01010,
                0b01010,
                0b01010,
                0b01110,
                0b11111,
                0b11111,
                0b01110
            ]
            
            # Water drop symbol
            water_char = [
                0b00100,
                0b00100,
                0b01010,
                0b01010,
                0b10001,
                0b10001,
                0b01110,
                0b00000
            ]
            
            # Plant symbol
            plant_char = [
                0b00100,
                0b01110,
                0b11111,
                0b00100,
                0b00100,
                0b01110,
                0b11111,
                0b00000
            ]
            
            # Create characters
            self.lcd.create_char(0, temp_char)
            self.lcd.create_char(1, water_char)
            self.lcd.create_char(2, plant_char)
            
            self.logger.info("Custom LCD characters created")
            
        except Exception as e:
            self.logger.error(f"Custom character creation error: {e}")
    
    def display_with_custom_chars(self, temp, humidity, soil):
        """Display using custom characters"""
        if not self.lcd:
            return
        
        try:
            with self.display_lock:
                self.lcd.clear()
                
                # Line 1: Temperature with custom char
                line1 = f"\x00{temp:4.1f}C \x01{humidity:2.0f}%"
                self.lcd.write_string(line1)
                
                # Line 2: Soil with custom char
                self.lcd.crlf()
                line2 = f"\x02{soil:2.0f}%"
                self.lcd.write_string(line2)
                
        except Exception as e:
            self.logger.error(f"Custom character display error: {e}")
    
    def cleanup(self):
        """Cleanup LCD resources"""
        try:
            if self.lcd:
                self.lcd.clear()
                self.show_message("BloomBotanics", "Shutdown...")
                time.sleep(1)
                self.lcd.clear()
                self.set_backlight(False)
                self.logger.info("LCD cleaned up")
        except Exception as e:
            self.logger.error(f"LCD cleanup error: {e}")
