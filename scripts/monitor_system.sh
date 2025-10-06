#!/bin/bash
"""
BloomBotanics System Monitor
Real-time monitoring with alerts and status display
"""

set -e

PROJECT_DIR="/home/pi/BloomBotanics"
MONITOR_INTERVAL=10  # seconds
ALERT_LOG="$PROJECT_DIR/data/logs/monitor_alerts.log"

# Colors and formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m'
BOLD='\033[1m'

# Create alert log
mkdir -p "$(dirname "$ALERT_LOG")"
touch "$ALERT_LOG"

# Function to log alerts
log_alert() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$ALERT_LOG"
}

# Function to get service status
get_service_status() {
    if systemctl is-active --quiet "$1"; then
        echo "🟢 RUNNING"
    else
        echo "🔴 STOPPED"
    fi
}

# Function to get temperature with color coding
get_temp_status() {
    local temp=$1
    if (( $(echo "$temp > 80" | bc -l) )); then
        echo -e "${RED}🔥 $temp°C${NC}"
    elif (( $(echo "$temp > 70" | bc -l) )); then
        echo -e "${YELLOW}🌡️ $temp°C${NC}"
    else
        echo -e "${GREEN}❄️ $temp°C${NC}"
    fi
}

# Function to get memory status with color coding
get_memory_status() {
    local usage=$1
    if (( $(echo "$usage > 90" | bc -l) )); then
        echo -e "${RED}🔴 $usage%${NC}"
    elif (( $(echo "$usage > 80" | bc -l) )); then
        echo -e "${YELLOW}🟡 $usage%${NC}"
    else
        echo -e "${GREEN}🟢 $usage%${NC}"
    fi
}

# Function to get disk status with color coding
get_disk_status() {
    local usage=$1
    if (( $(echo "$usage > 90" | bc -l) )); then
        echo -e "${RED}💾 $usage%${NC}"
    elif (( $(echo "$usage > 80" | bc -l) )); then
        echo -e "${YELLOW}💾 $usage%${NC}"
    else
        echo -e "${GREEN}💾 $usage%${NC}"
    fi
}

# Function to check and alert on issues
check_alerts() {
    local cpu_temp=$(cat /sys/class/thermal/thermal_zone0/temp | awk '{print $1/1000}')
    local memory_usage=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')
    local disk_usage=$(df / | awk 'NR==2{print $(NF-1)}' | sed 's/%//')
    
    # Temperature alerts
    if (( $(echo "$cpu_temp > 80" | bc -l) )); then
        log_alert "CRITICAL: CPU temperature $cpu_temp°C"
    fi
    
    # Memory alerts
    if (( $(echo "$memory_usage > 95" | bc -l) )); then
        log_alert "CRITICAL: Memory usage ${memory_usage}%"
    fi
    
    # Disk alerts
    if (( $(echo "$disk_usage > 95" | bc -l) )); then
        log_alert "CRITICAL: Disk usage ${disk_usage}%"
    fi
    
    # Service alerts
    if ! systemctl is-active --quiet bloom-botanics; then
        log_alert "CRITICAL: BloomBotanics service is not running"
    fi
}

# Function to display header
display_header() {
    echo -e "${BOLD}${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                          🌱 BloomBotanics Monitor                           ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "${WHITE}System Time: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${WHITE}Monitoring Interval: ${MONITOR_INTERVAL}s${NC}"
    echo ""
}

# Function to display system status
display_system_status() {
    # Get system metrics
    local cpu_temp=$(cat /sys/class/thermal/thermal_zone0/temp | awk '{print $1/1000}')
    local memory_info=$(free -h | awk 'NR==2{printf "%s/%s (%.1f%%)", $3, $2, $3*100/$2}')
    local memory_usage=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')
    local disk_info=$(df -h / | awk 'NR==2{printf "%s/%s (%s)", $3, $2, $5}')
    local disk_usage=$(df / | awk 'NR==2{print $(NF-1)}' | sed 's/%//')
    local load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1 $2 $3}')
    local uptime_info=$(uptime -p)
    
    echo -e "${BOLD}${PURPLE}═══ System Status ═══${NC}"
    echo -e "${CYAN}🌡️  CPU Temperature:${NC} $(get_temp_status $cpu_temp)"
    echo -e "${CYAN}🧠 Memory Usage:${NC} $(get_memory_status $memory_usage) ($memory_info)"
    echo -e "${CYAN}💾 Disk Usage:${NC} $(get_disk_status $disk_usage) ($disk_info)"
    echo -e "${CYAN}⚡ Load Average:${NC} $load_avg"
    echo -e "${CYAN}⏱️  Uptime:${NC} $uptime_info"
    echo ""
}

# Function to display service status
display_service_status() {
    echo -e "${BOLD}${PURPLE}═══ Service Status ═══${NC}"
    
    services=("bloom-botanics" "ssh" "rpi-connect")
    
    for service in "${services[@]}"; do
        local status=$(get_service_status "$service")
        local memory_usage=""
        
        if systemctl is-active --quiet "$service"; then
            # Get memory usage for running service
            local pid=$(systemctl show --property MainPID --value "$service" 2>/dev/null)
            if [ -n "$pid" ] && [ "$pid" != "0" ]; then
                memory_usage=$(ps -p "$pid" -o rss= 2>/dev/null | awk '{printf "%.1fMB", $1/1024}' || echo "N/A")
            fi
        fi
        
        echo -e "${CYAN}🔧 $service:${NC} $status ${memory_usage:+($memory_usage)}"
    done
    echo ""
}

# Function to display network status
display_network_status() {
    echo -e "${BOLD}${PURPLE}═══ Network Status ═══${NC}"
    
    # Check connectivity
    local internet_status="🔴 OFFLINE"
    if ping -c 1 -W 2 8.8.8.8 >/dev/null 2>&1; then
        internet_status="🟢 ONLINE"
    fi
    
    # Get IP addresses
    local eth_ip=$(ip route get 8.8.8.8 2>/dev/null | awk '{print $7; exit}' || echo "N/A")
    local wlan_ip=$(ip addr show wlan0 2>/dev/null | grep 'inet ' | awk '{print $2}' | cut -d/ -f1 || echo "N/A")
    
    echo -e "${CYAN}🌐 Internet:${NC} $internet_status"
    echo -e "${CYAN}🔌 Ethernet IP:${NC} $eth_ip"
    echo -e "${CYAN}📶 WiFi IP:${NC} $wlan_ip"
    
    # Pi Connect status
    local pi_connect_status="🔴 INACTIVE"
    if systemctl is-active --quiet rpi-connect; then
        pi_connect_status="🟢 ACTIVE"
    fi
    echo -e "${CYAN}🔗 Pi Connect:${NC} $pi_connect_status"
    echo ""
}

# Function to display BloomBotanics specific status
display_bloom_status() {
    echo -e "${BOLD}${PURPLE}═══ BloomBotanics Status ═══${NC}"
    
    # Check if config exists and get phone number
    local phone_number="Not configured"
    if [ -f "$PROJECT_DIR/config.py" ]; then
        phone_number=$(grep "PHONE_NUMBER" "$PROJECT_DIR/config.py" | head -1 | cut -d'"' -f2 || echo "Not configured")
    fi
    
    # Count data files
    local sensor_files=$(find "$PROJECT_DIR/data/sensor_data" -name "*.json" 2>/dev/null | wc -l)
    local image_files=$(find "$PROJECT_DIR/data/images" -name "*.jpg" 2>/dev/null | wc -l)
    local detection_files=$(find "$PROJECT_DIR/data/detections" -name "*.jpg" 2>/dev/null | wc -l)
    
    # Latest log entries
    local latest_log=""
    if [ -f "$PROJECT_DIR/data/logs/bloom_botanics.log" ]; then
        latest_log=$(tail -1 "$PROJECT_DIR/data/logs/bloom_botanics.log" 2>/dev/null | cut -d' ' -f1-3 || echo "No logs")
    fi
    
    echo -e "${CYAN}📱 SMS Number:${NC} $phone_number"
    echo -e "${CYAN}📊 Sensor Files:${NC} $sensor_files"
    echo -e "${CYAN}📸 Images:${NC} $image_files"
    echo -e "${CYAN}🚨 Detections:${NC} $detection_files"
    echo -e "${CYAN}📝 Latest Log:${NC} $latest_log"
    echo ""
}

# Function to display recent alerts
display_recent_alerts() {
    echo -e "${BOLD}${PURPLE}═══ Recent Alerts ═══${NC}"
    
    if [ -f "$ALERT_LOG" ] && [ -s "$ALERT_LOG" ]; then
        local alert_count=$(tail -10 "$ALERT_LOG" | wc -l)
        echo -e "${CYAN}Recent alerts (last 10):${NC}"
        tail -10 "$ALERT_LOG" | while read -r line; do
            echo -e "${YELLOW}⚠️ $line${NC}"
        done
    else
        echo -e "${GREEN}✅ No recent alerts${NC}"
    fi
    echo ""
}

# Function to display quick stats
display_quick_stats() {
    local runtime=$(systemctl show --property ActiveEnterTimestamp --value bloom-botanics 2>/dev/null)
    local restart_count=$(systemctl show --property NRestarts --value bloom-botanics 2>/dev/null)
    
    echo -e "${BOLD}${PURPLE}═══ Quick Stats ═══${NC}"
    echo -e "${CYAN}🔄 Service Restarts:${NC} ${restart_count:-0}"
    echo -e "${CYAN}⏰ Last Started:${NC} ${runtime:-Unknown}"
    
    # Data directory sizes
    local data_size=$(du -sh "$PROJECT_DIR/data" 2>/dev/null | cut -f1 || echo "N/A")
    echo -e "${CYAN}💾 Data Directory:${NC} $data_size"
    echo ""
}

# Function for interactive mode
interactive_mode() {
    echo -e "${GREEN}🔄 Starting interactive monitoring mode...${NC}"
    echo -e "${YELLOW}Press 'q' to quit, 'r' to refresh, 'c' to clear alerts${NC}"
    echo ""
    
    while true; do
        clear
        display_header
        display_system_status
        display_service_status
        display_network_status
        display_bloom_status
        display_quick_stats
        display_recent_alerts
        check_alerts
        
        echo -e "${WHITE}Commands: [q]uit | [r]efresh | [c]lear alerts | Auto-refresh in ${MONITOR_INTERVAL}s${NC}"
        
        # Read input with timeout
        if read -t $MONITOR_INTERVAL -n 1 input; then
            case $input in
                q|Q)
                    echo -e "\n${GREEN}Monitoring stopped.${NC}"
                    exit 0
                    ;;
                r|R)
                    continue
                    ;;
                c|C)
                    > "$ALERT_LOG"
                    echo -e "\n${GREEN}Alerts cleared.${NC}"
                    sleep 1
                    ;;
            esac
        fi
    done
}

# Function for single check mode
single_check_mode() {
    display_header
    display_system_status
    display_service_status  
    display_network_status
    display_bloom_status
    display_quick_stats
    display_recent_alerts
    check_alerts
}

# Function to show help
show_help() {
    echo "BloomBotanics System Monitor"
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -i, --interactive    Interactive monitoring mode (default)"
    echo "  -s, --single        Single check mode"
    echo "  -h, --help          Show this help"
    echo "  -t, --interval N    Set monitoring interval (seconds, default: 10)"
    echo ""
    echo "Examples:"
    echo "  $0                  # Interactive monitoring"
    echo "  $0 -s              # Single check"
    echo "  $0 -i -t 5         # Interactive with 5s interval"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--interactive)
            MODE="interactive"
            shift
            ;;
        -s|--single)
            MODE="single"
            shift
            ;;
        -t|--interval)
            MONITOR_INTERVAL="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Default to interactive mode
MODE=${MODE:-interactive}

# Main execution
case $MODE in
    interactive)
        interactive_mode
        ;;
    single)
        single_check_mode
        ;;
esac
