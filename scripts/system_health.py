#!/usr/bin/env python3
"""
BloomBotanics System Health Monitor
Comprehensive system health checking and reporting
"""

import os
import sys
import time
import json
import subprocess
import psutil
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import *
    from utils.logger import get_logger
except ImportError:
    print("Warning: Could not import BloomBotanics modules")
    LOG_DIR = "/home/pi/BloomBotanics/data/logs"

class SystemHealthMonitor:
    def __init__(self):
        self.logger = get_logger(__name__) if 'get_logger' in globals() else None
        self.health_data = {}
        
    def log_info(self, message):
        """Log message with fallback if logger not available"""
        if self.logger:
            self.logger.info(message)
        else:
            print(f"[INFO] {message}")
    
    def log_warning(self, message):
        """Log warning with fallback"""
        if self.logger:
            self.logger.warning(message)
        else:
            print(f"[WARNING] {message}")
    
    def log_error(self, message):
        """Log error with fallback"""
        if self.logger:
            self.logger.error(message)
        else:
            print(f"[ERROR] {message}")

    def check_cpu_health(self):
        """Check CPU temperature, usage, and frequency"""
        try:
            # CPU Temperature
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                cpu_temp = float(f.read().strip()) / 1000.0
            
            # CPU Usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # CPU Frequency
            try:
                cpu_freq = psutil.cpu_freq()
                current_freq = cpu_freq.current if cpu_freq else 0
            except:
                current_freq = 0
            
            # Load Average
            load_avg = os.getloadavg()
            
            cpu_health = {
                'temperature': cpu_temp,
                'usage_percent': cpu_usage,
                'frequency_mhz': current_freq,
                'load_average_1min': load_avg[0],
                'load_average_5min': load_avg[1],
                'load_average_15min': load_avg[2]
            }
            
            # Health assessment
            issues = []
            if cpu_temp > 80:
                issues.append(f"Critical CPU temperature: {cpu_temp:.1f}°C")
            elif cpu_temp > 70:
                issues.append(f"High CPU temperature: {cpu_temp:.1f}°C")
            
            if cpu_usage > 90:
                issues.append(f"Very high CPU usage: {cpu_usage:.1f}%")
            elif cpu_usage > 80:
                issues.append(f"High CPU usage: {cpu_usage:.1f}%")
            
            cpu_health['issues'] = issues
            cpu_health['status'] = 'critical' if any('Critical' in issue for issue in issues) else \
                                   'warning' if issues else 'healthy'
            
            return cpu_health
            
        except Exception as e:
            self.log_error(f"CPU health check failed: {e}")
            return {'status': 'error', 'error': str(e)}

    def check_memory_health(self):
        """Check RAM and swap usage"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            memory_health = {
                'total_mb': memory.total // 1024 // 1024,
                'used_mb': memory.used // 1024 // 1024,
                'free_mb': memory.free // 1024 // 1024,
                'usage_percent': memory.percent,
                'swap_total_mb': swap.total // 1024 // 1024,
                'swap_used_mb': swap.used // 1024 // 1024,
                'swap_percent': swap.percent
            }
            
            issues = []
            if memory.percent > 95:
                issues.append(f"Critical memory usage: {memory.percent:.1f}%")
            elif memory.percent > 85:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            
            if swap.percent > 80:
                issues.append(f"High swap usage: {swap.percent:.1f}%")
            
            memory_health['issues'] = issues
            memory_health['status'] = 'critical' if any('Critical' in issue for issue in issues) else \
                                     'warning' if issues else 'healthy'
            
            return memory_health
            
        except Exception as e:
            self.log_error(f"Memory health check failed: {e}")
            return {'status': 'error', 'error': str(e)}

    def check_disk_health(self):
        """Check disk space and I/O"""
        try:
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            
            disk_health = {
                'total_gb': disk.total // 1024 // 1024 // 1024,
                'used_gb': disk.used // 1024 // 1024 // 1024,
                'free_gb': disk.free // 1024 // 1024 // 1024,
                'usage_percent': (disk.used / disk.total) * 100,
                'read_count': disk_io.read_count if disk_io else 0,
                'write_count': disk_io.write_count if disk_io else 0,
                'read_bytes': disk_io.read_bytes if disk_io else 0,
                'write_bytes': disk_io.write_bytes if disk_io else 0
            }
            
            issues = []
            usage_percent = disk_health['usage_percent']
            
            if usage_percent > 95:
                issues.append(f"Critical disk usage: {usage_percent:.1f}%")
            elif usage_percent > 90:
                issues.append(f"High disk usage: {usage_percent:.1f}%")
            elif usage_percent > 80:
                issues.append(f"Moderate disk usage: {usage_percent:.1f}%")
            
            # Check for SD card wear (read/write ratio)
            if disk_io and disk_io.write_count > 0:
                rw_ratio = disk_io.read_count / disk_io.write_count
                if rw_ratio < 1.0:  # More writes than reads (concerning for SD cards)
                    issues.append(f"High SD card write activity (R/W ratio: {rw_ratio:.2f})")
            
            disk_health['issues'] = issues
            disk_health['status'] = 'critical' if any('Critical' in issue for issue in issues) else \
                                   'warning' if issues else 'healthy'
            
            return disk_health
            
        except Exception as e:
            self.log_error(f"Disk health check failed: {e}")
            return {'status': 'error', 'error': str(e)}

    def check_network_health(self):
        """Check network connectivity and interface status"""
        try:
            network_health = {
                'interfaces': {},
                'connectivity': {}
            }
            
            # Network interfaces
            for interface, addresses in psutil.net_if_addrs().items():
                if_stats = psutil.net_if_stats().get(interface)
                
                network_health['interfaces'][interface] = {
                    'is_up': if_stats.isup if if_stats else False,
                    'addresses': [addr.address for addr in addresses],
                    'speed_mbps': if_stats.speed if if_stats else 0
                }
            
            # Network I/O
            net_io = psutil.net_io_counters()
            if net_io:
                network_health['io'] = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv,
                    'errors_in': net_io.errin,
                    'errors_out': net_io.errout
                }
            
            # Connectivity tests
            connectivity_tests = [
                ('Google DNS', '8.8.8.8'),
                ('Cloudflare DNS', '1.1.1.1'),
                ('Local Gateway', self._get_default_gateway())
            ]
            
            for name, host in connectivity_tests:
                if host:
                    network_health['connectivity'][name] = self._ping_test(host)
                else:
                    network_health['connectivity'][name] = False
            
            # Assess network health
            issues = []
            active_interfaces = [name for name, info in network_health['interfaces'].items() 
                               if info['is_up'] and name != 'lo']
            
            if not active_interfaces:
                issues.append("No active network interfaces")
            
            if not any(network_health['connectivity'].values()):
                issues.append("No internet connectivity")
            
            if net_io and net_io.errin > 100:
                issues.append(f"Network input errors: {net_io.errin}")
            
            network_health['issues'] = issues
            network_health['status'] = 'critical' if not active_interfaces else \
                                     'warning' if issues else 'healthy'
            
            return network_health
            
        except Exception as e:
            self.log_error(f"Network health check failed: {e}")
            return {'status': 'error', 'error': str(e)}

    def check_service_health(self):
        """Check BloomBotanics service and related services"""
        try:
            services_to_check = [
                'bloom-botanics',
                'ssh',
                'rpi-connect'
            ]
            
            service_health = {}
            
            for service in services_to_check:
                try:
                    result = subprocess.run(
                        ['systemctl', 'is-active', service],
                        capture_output=True,
                        text=True
                    )
                    
                    is_active = result.returncode == 0
                    status = result.stdout.strip() if result.stdout else 'unknown'
                    
                    # Get more detailed info if service is running
                    if is_active:
                        status_result = subprocess.run(
                            ['systemctl', 'status', service, '--no-pager', '-l'],
                            capture_output=True,
                            text=True
                        )
                        
                        # Extract memory usage and runtime
                        memory_usage = 0
                        runtime = "unknown"
                        
                        for line in status_result.stdout.split('\n'):
                            if 'Memory:' in line:
                                try:
                                    memory_usage = float(line.split('Memory:')[1].strip().split()[0])
                                except:
                                    pass
                            elif 'Active:' in line and 'since' in line:
                                try:
                                    runtime = line.split('since')[1].strip()
                                except:
                                    pass
                    
                    service_health[service] = {
                        'active': is_active,
                        'status': status,
                        'memory_mb': memory_usage,
                        'runtime': runtime
                    }
                    
                except Exception as e:
                    service_health[service] = {
                        'active': False,
                        'status': 'error',
                        'error': str(e)
                    }
            
            # Overall service health assessment
            issues = []
            critical_services = ['bloom-botanics']
            
            for service in critical_services:
                if service in service_health and not service_health[service]['active']:
                    issues.append(f"Critical service not running: {service}")
            
            service_health['issues'] = issues
            service_health['status'] = 'critical' if any('Critical' in issue for issue in issues) else \
                                     'warning' if issues else 'healthy'
            
            return service_health
            
        except Exception as e:
            self.log_error(f"Service health check failed: {e}")
            return {'status': 'error', 'error': str(e)}

    def check_hardware_health(self):
        """Check hardware-specific health indicators"""
        try:
            hardware_health = {}
            
            # GPU temperature (if available)
            try:
                result = subprocess.run(
                    ['vcgencmd', 'measure_temp'],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    temp_str = result.stdout.strip()
                    gpu_temp = float(temp_str.split('=')[1].split("'")[0])
                    hardware_health['gpu_temperature'] = gpu_temp
            except:
                hardware_health['gpu_temperature'] = None
            
            # Voltage readings
            voltage_commands = {
                'core': 'measure_volts core',
                'sdram_c': 'measure_volts sdram_c',
                'sdram_i': 'measure_volts sdram_i',
                'sdram_p': 'measure_volts sdram_p'
            }
            
            hardware_health['voltages'] = {}
            for name, command in voltage_commands.items():
                try:
                    result = subprocess.run(
                        ['vcgencmd', command],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        voltage_str = result.stdout.strip()
                        voltage = float(voltage_str.split('=')[1].split('V')[0])
                        hardware_health['voltages'][name] = voltage
                except:
                    hardware_health['voltages'][name] = None
            
            # Throttling check
            try:
                result = subprocess.run(
                    ['vcgencmd', 'get_throttled'],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    throttled_hex = result.stdout.strip().split('=')[1]
                    throttled_int = int(throttled_hex, 16)
                    
                    hardware_health['throttling'] = {
                        'raw_value': throttled_hex,
                        'under_voltage_detected': bool(throttled_int & 0x1),
                        'arm_frequency_capped': bool(throttled_int & 0x2),
                        'currently_throttled': bool(throttled_int & 0x4),
                        'soft_temperature_limit': bool(throttled_int & 0x8)
                    }
            except:
                hardware_health['throttling'] = None
            
            # Assess hardware issues
            issues = []
            
            if hardware_health.get('gpu_temperature', 0) > 85:
                issues.append(f"High GPU temperature: {hardware_health['gpu_temperature']:.1f}°C")
            
            # Check voltages
            core_voltage = hardware_health['voltages'].get('core')
            if core_voltage and core_voltage < 1.2:
                issues.append(f"Low core voltage: {core_voltage:.2f}V")
            
            # Check throttling
            throttling = hardware_health.get('throttling', {})
            if throttling:
                if throttling.get('under_voltage_detected'):
                    issues.append("Under-voltage detected (power supply issue)")
                if throttling.get('currently_throttled'):
                    issues.append("System currently throttled")
                if throttling.get('soft_temperature_limit'):
                    issues.append("Soft temperature limit reached")
            
            hardware_health['issues'] = issues
            hardware_health['status'] = 'critical' if any('voltage' in issue.lower() for issue in issues) else \
                                      'warning' if issues else 'healthy'
            
            return hardware_health
            
        except Exception as e:
            self.log_error(f"Hardware health check failed: {e}")
            return {'status': 'error', 'error': str(e)}

    def _get_default_gateway(self):
        """Get default gateway IP"""
        try:
            result = subprocess.run(['ip', 'route', 'show', 'default'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'default via' in line:
                    return line.split('default via')[1].split()[0]
            return None
        except:
            return None

    def _ping_test(self, host, count=2):
        """Test network connectivity to host"""
        try:
            result = subprocess.run(
                ['ping', '-c', str(count), '-W', '3', host],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except:
            return False

    def generate_health_report(self):
        """Generate comprehensive health report"""
        print("🏥 BloomBotanics System Health Report")
        print("=====================================")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Run all health checks
        health_checks = {
            'CPU': self.check_cpu_health(),
            'Memory': self.check_memory_health(),
            'Disk': self.check_disk_health(),
            'Network': self.check_network_health(),
            'Services': self.check_service_health(),
            'Hardware': self.check_hardware_health()
        }
        
        # Overall system status
        all_statuses = [check.get('status', 'unknown') for check in health_checks.values()]
        overall_status = 'critical' if 'critical' in all_statuses else \
                        'warning' if 'warning' in all_statuses else 'healthy'
        
        status_emoji = "🔴" if overall_status == 'critical' else \
                      "🟡" if overall_status == 'warning' else "🟢"
        
        print(f"Overall System Status: {status_emoji} {overall_status.upper()}")
        print()
        
        # Detailed checks
        for category, health_data in health_checks.items():
            status = health_data.get('status', 'unknown')
            emoji = "🔴" if status == 'critical' else \
                   "🟡" if status == 'warning' else \
                   "🟢" if status == 'healthy' else "⚪"
            
            print(f"{emoji} {category}: {status.upper()}")
            
            # Show key metrics
            if category == 'CPU':
                temp = health_data.get('temperature', 0)
                usage = health_data.get('usage_percent', 0)
                print(f"   Temperature: {temp:.1f}°C, Usage: {usage:.1f}%")
            
            elif category == 'Memory':
                usage = health_data.get('usage_percent', 0)
                used = health_data.get('used_mb', 0)
                total = health_data.get('total_mb', 0)
                print(f"   Usage: {usage:.1f}% ({used}MB / {total}MB)")
            
            elif category == 'Disk':
                usage = health_data.get('usage_percent', 0)
                free = health_data.get('free_gb', 0)
                total = health_data.get('total_gb', 0)
                print(f"   Usage: {usage:.1f}% ({free}GB free / {total}GB total)")
            
            elif category == 'Network':
                interfaces = health_data.get('interfaces', {})
                active = [name for name, info in interfaces.items() if info.get('is_up') and name != 'lo']
                connectivity = health_data.get('connectivity', {})
                online = sum(1 for connected in connectivity.values() if connected)
                print(f"   Active interfaces: {len(active)}, Connectivity: {online}/{len(connectivity)} tests passed")
            
            elif category == 'Services':
                services = [name for name, info in health_data.items() 
                           if isinstance(info, dict) and info.get('active')]
                print(f"   Running services: {len(services)}")
            
            elif category == 'Hardware':
                gpu_temp = health_data.get('gpu_temperature')
                throttling = health_data.get('throttling', {})
                if gpu_temp:
                    print(f"   GPU Temperature: {gpu_temp:.1f}°C")
                if throttling and throttling.get('currently_throttled'):
                    print(f"   WARNING: System is currently throttled")
            
            # Show issues
            issues = health_data.get('issues', [])
            for issue in issues:
                print(f"   ⚠️ {issue}")
            
            print()
        
        # Save health report to file
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'health_checks': health_checks,
            'system_info': {
                'uptime': time.time() - psutil.boot_time(),
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                'python_version': sys.version,
                'platform': os.uname()._asdict() if hasattr(os.uname(), '_asdict') else str(os.uname())
            }
        }
        
        # Save to file
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            report_file = os.path.join(LOG_DIR, f'health_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"📄 Detailed report saved to: {report_file}")
        except Exception as e:
            print(f"❌ Could not save report: {e}")
        
        return report_data

def main():
    """Main function"""
    monitor = SystemHealthMonitor()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'cpu':
            result = monitor.check_cpu_health()
            print(json.dumps(result, indent=2))
        elif command == 'memory':
            result = monitor.check_memory_health()
            print(json.dumps(result, indent=2))
        elif command == 'disk':
            result = monitor.check_disk_health()
            print(json.dumps(result, indent=2))
        elif command == 'network':
            result = monitor.check_network_health()
            print(json.dumps(result, indent=2))
        elif command == 'services':
            result = monitor.check_service_health()
            print(json.dumps(result, indent=2))
        elif command == 'hardware':
            result = monitor.check_hardware_health()
            print(json.dumps(result, indent=2))
        else:
            print("Usage: python3 system_health.py [cpu|memory|disk|network|services|hardware]")
            print("       python3 system_health.py  (for full report)")
    else:
        # Full health report
        monitor.generate_health_report()

if __name__ == "__main__":
    main()
