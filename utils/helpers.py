"""Helper utilities for system health and data management"""

import os
import json
import psutil
import time
from datetime import datetime, timedelta
from config import *

class SystemHealth:
    """Monitor system health metrics"""
    
    def get_system_health(self):
        """Get comprehensive system health status"""
        try:
            # CPU temperature
            cpu_temp = self._get_cpu_temperature()
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Network status
            network_connected = self._check_network()
            
            # Uptime
            uptime = time.time() - psutil.boot_time()
            
            return {
                'cpu_temp': cpu_temp,
                'memory_usage': memory_usage,
                'disk_usage': disk_usage,
                'cpu_usage': cpu_usage,
                'network_connected': network_connected,
                'uptime_hours': uptime / 3600,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_cpu_temperature(self):
        """Get CPU temperature"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read()) / 1000.0
            return temp
        except:
            return 0.0
    
    def _check_network(self):
        """Check network connectivity"""
        try:
            import subprocess
            result = subprocess.run(['ping', '-c', '1', '8.8.8.8'], 
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False

class DataManager:
    """Manage sensor data storage and retrieval"""
    
    def __init__(self):
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure all data directories exist"""
        for directory in [SENSOR_DATA_DIR, IMAGE_DIR, DETECTION_DIR, BACKUP_DIR]:
            os.makedirs(directory, exist_ok=True)
    
    def save_sensor_data(self, data):
        """Save sensor data to daily JSON file"""
        try:
            date_str = datetime.now().strftime('%Y%m%d')
            filename = os.path.join(SENSOR_DATA_DIR, f'sensors_{date_str}.json')
            
            with open(filename, 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            print(f"Data save error: {e}")
    
    def get_daily_statistics(self, date=None):
        """Get daily statistics for reporting"""
        if not date:
            date = datetime.now().date()
        
        date_str = date.strftime('%Y%m%d')
        filename = os.path.join(SENSOR_DATA_DIR, f'sensors_{date_str}.json')
        
        if not os.path.exists(filename):
            return {}
        
        try:
            temperatures = []
            humidities = []
            irrigation_count = 0
            threat_count = 0
            
            with open(filename, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    
                    if data.get('temperature'):
                        temperatures.append(data['temperature'])
                    if data.get('humidity'):
                        humidities.append(data['humidity'])
                    
                    # Count irrigation events (placeholder)
                    # You can enhance this based on your data structure
            
            return {
                'avg_temperature': sum(temperatures) / len(temperatures) if temperatures else 0,
                'avg_humidity': sum(humidities) / len(humidities) if humidities else 0,
                'irrigation_count': irrigation_count,
                'threat_count': threat_count,
                'data_points': len(temperatures)
            }
        except Exception as e:
            print(f"Statistics error: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep=BACKUP_RETENTION_DAYS):
        """Clean up old data files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            for directory in [SENSOR_DATA_DIR, IMAGE_DIR, DETECTION_DIR]:
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    if os.path.isfile(file_path):
                        file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                        if file_time < cutoff_date:
                            os.remove(file_path)
        except Exception as e:
            print(f"Cleanup error: {e}")
