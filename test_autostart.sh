#!/bin/bash
"""
BloomBotanics Auto-Start Verification Script
Tests if the system will start automatically on boot
"""

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[CHECK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }

echo "🔧 BloomBotanics Auto-Start Verification"
echo "========================================"
echo "Testing auto-start configuration..."
echo ""

# Initialize counters
CHECKS_PASSED=0
TOTAL_CHECKS=8
ISSUES=()

# Check 1: Service file exists
print_info "1. Checking systemd service file..."
if [ -f "/etc/systemd/system/bloom-botanics.service" ]; then
    print_status "✅ Service file exists"
    ((CHECKS_PASSED++))
else
    print_error "❌ Service file missing: /etc/systemd/system/bloom-botanics.service"
    ISSUES+=("Service file not installed")
fi

# Check 2: Service is enabled
print_info "2. Checking if service is enabled for auto-start..."
if systemctl is-enabled bloom-botanics.service | grep -q "enabled"; then
    print_status "✅ Service enabled for auto-start"
    ((CHECKS_PASSED++))
else
    print_error "❌ Service NOT enabled for auto-start"
    print_error "   Fix: sudo systemctl enable bloom-botanics.service"
    ISSUES+=("Service not enabled")
fi

# Check 3: Main Python file exists and is executable
print_info "3. Checking main Python file..."
if [ -f "/home/pi/BloomBotanics/main.py" ]; then
    if [ -x "/home/pi/BloomBotanics/main.py" ]; then
        print_status "✅ main.py exists and is executable"
        ((CHECKS_PASSED++))
    else
        print_warning "⚠️ main.py exists but not executable"
        print_info "   Fix: chmod +x /home/pi/BloomBotanics/main.py"
        ISSUES+=("main.py not executable")
    fi
else
    print_error "❌ main.py not found"
    ISSUES+=("main.py missing")
fi

# Check 4: Configuration file exists
print_info "4. Checking configuration file..."
if [ -f "/home/pi/BloomBotanics/config.py" ]; then
    print_status "✅ config.py exists"
    
    # Check for default phone number
    if grep -q "+919876543210" "/home/pi/BloomBotanics/config.py"; then
        print_warning "⚠️ Phone number still set to default"
        print_info "   Update PHONE_NUMBER in config.py"
        ISSUES+=("Default phone number")
    else
        print_status "✅ Phone number appears to be configured"
    fi
    ((CHECKS_PASSED++))
else
    print_error "❌ config.py not found"
    ISSUES+=("config.py missing")
fi

# Check 5: Data directories exist
print_info "5. Checking data directories..."
REQUIRED_DIRS=(
    "/home/pi/BloomBotanics/data"
    "/home/pi/BloomBotanics/data/sensor_data"
    "/home/pi/BloomBotanics/data/images"
    "/home/pi/BloomBotanics/data/detections"
    "/home/pi/BloomBotanics/data/logs"
)

missing_dirs=0
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        print_warning "⚠️ Missing directory: $dir"
        ((missing_dirs++))
    fi
done

if [ $missing_dirs -eq 0 ]; then
    print_status "✅ All data directories exist"
    ((CHECKS_PASSED++))
else
    print_error "❌ $missing_dirs data directories missing"
    print_info "   Fix: mkdir -p /home/pi/BloomBotanics/data/{sensor_data,images,detections,logs}"
    ISSUES+=("Missing data directories")
fi

# Check 6: Hardware interfaces enabled
print_info "6. Checking hardware interfaces..."
interface_issues=0

# Check camera
if ! raspi-config nonint get_camera | grep -q "0"; then
    print_warning "⚠️ Camera interface not enabled"
    ((interface_issues++))
fi

# Check I2C
if ! raspi-config nonint get_i2c | grep -q "0"; then
    print_warning "⚠️ I2C interface not enabled"
    ((interface_issues++))
fi

# Check SPI
if ! raspi-config nonint get_spi | grep -q "0"; then
    print_warning "⚠️ SPI interface not enabled"
    ((interface_issues++))
fi

if [ $interface_issues -eq 0 ]; then
    print_status "✅ All hardware interfaces enabled"
    ((CHECKS_PASSED++))
else
    print_error "❌ $interface_issues hardware interfaces not enabled"
    print_info "   Fix: sudo raspi-config -> Interface Options"
    ISSUES+=("Hardware interfaces not enabled")
fi

# Check 7: Python dependencies
print_info "7. Checking Python dependencies..."
REQUIRED_PACKAGES=("RPi.GPIO" "adafruit-circuitpython-dht" "picamera2" "RPLCD" "pyserial")
missing_packages=0

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python3 -c "import $package" 2>/dev/null; then
        print_warning "⚠️ Missing Python package: $package"
        ((missing_packages++))
    fi
done

if [ $missing_packages -eq 0 ]; then
    print_status "✅ All required Python packages available"
    ((CHECKS_PASSED++))
else
    print_error "❌ $missing_packages Python packages missing"
    print_info "   Fix: pip install --break-system-packages -r requirements.txt"
    ISSUES+=("Missing Python packages")
fi

# Check 8: Current service status
print_info "8. Checking current service status..."
SERVICE_STATUS=$(systemctl is-active bloom-botanics.service 2>/dev/null || echo "inactive")

case $SERVICE_STATUS in
    "active")
        print_status "✅ Service is currently running"
        ((CHECKS_PASSED++))
        
        # Get additional info
        MEMORY_USAGE=$(systemctl show --property MemoryCurrent --value bloom-botanics 2>/dev/null)
        if [ -n "$MEMORY_USAGE" ] && [ "$MEMORY_USAGE" != "0" ]; then
            MEMORY_MB=$((MEMORY_USAGE / 1024 / 1024))
            print_info "   Memory usage: ${MEMORY_MB}MB"
        fi
        
        UPTIME=$(systemctl show --property ActiveEnterTimestamp --value bloom-botanics 2>/dev/null)
        if [ -n "$UPTIME" ]; then
            print_info "   Started: $UPTIME"
        fi
        ;;
    "inactive")
        print_warning "⚠️ Service is not currently running"
        print_info "   This is normal if you haven't started it yet"
        print_info "   Start with: sudo systemctl start bloom-botanics"
        ;;
    "failed")
        print_error "❌ Service is in failed state"
        print_info "   Check logs: sudo journalctl -u bloom-botanics -n 20"
        ISSUES+=("Service in failed state")
        ;;
    *)
        print_warning "⚠️ Service status unknown: $SERVICE_STATUS"
        ;;
esac

# Pi Connect check (bonus)
print_info "Bonus: Checking Pi Connect..."
if systemctl is-enabled rpi-connect >/dev/null 2>&1; then
    if systemctl is-active rpi-connect >/dev/null 2>&1; then
        print_status "✅ Pi Connect is active"
    else
        print_warning "⚠️ Pi Connect enabled but not active"
    fi
else
    print_warning "⚠️ Pi Connect not enabled"
    print_info "   Enable: sudo systemctl enable rpi-connect"
fi

# Network connectivity check
print_info "Bonus: Checking network connectivity..."
if ping -c 1 -W 3 8.8.8.8 >/dev/null 2>&1; then
    print_status "✅ Internet connectivity available"
else
    print_warning "⚠️ No internet connectivity"
    print_info "   Check WiFi/Ethernet connection"
fi

echo ""
echo "📊 Auto-Start Verification Results"
echo "=================================="
echo "Checks Passed: $CHECKS_PASSED/$TOTAL_CHECKS"
echo "Success Rate: $(( (CHECKS_PASSED * 100) / TOTAL_CHECKS ))%"

if [ $CHECKS_PASSED -eq $TOTAL_CHECKS ]; then
    echo ""
    echo "🎉 EXCELLENT! Your BloomBotanics system is properly configured for auto-start!"
    echo ""
    echo "✅ The system will:"
    echo "   • Start automatically on every boot"
    echo "   • Resume operation after power outages"
    echo "   • Run continuously in the background"
    echo "   • Be accessible via Pi Connect"
    echo ""
    echo "🚀 Ready for deployment!"
    echo ""
    echo "Quick Commands:"
    echo "  Start now: sudo systemctl start bloom-botanics"
    echo "  View logs: sudo journalctl -u bloom-botanics -f"
    echo "  Check status: sudo systemctl status bloom-botanics"
    echo "  Test reboot: sudo reboot"
    echo ""
else
    echo ""
    echo "⚠️ Issues Found (${#ISSUES[@]}):"
    for issue in "${ISSUES[@]}"; do
        echo "  • $issue"
    done
    echo ""
    echo "🔧 Fix the issues above, then run this script again."
    echo ""
    echo "Common fixes:"
    echo "  Install service: sudo cp services/bloom-botanics.service /etc/systemd/system/"
    echo "  Enable service: sudo systemctl enable bloom-botanics"
    echo "  Create directories: mkdir -p data/{sensor_data,images,detections,logs}"
    echo "  Install packages: pip install --break-system-packages -r requirements.txt"
    echo "  Enable interfaces: sudo raspi-config"
    echo ""
fi

# Test auto-start simulation
echo "🧪 Auto-Start Simulation Test"
echo "============================="
echo ""
echo "To fully test auto-start:"
echo "1. Ensure all checks above pass"
echo "2. Reboot the system: sudo reboot"
echo "3. Wait 2-3 minutes after boot"
echo "4. Check if service started: sudo systemctl status bloom-botanics"
echo "5. Access via Pi Connect: https://connect.raspberrypi.com"
echo ""

if [ $CHECKS_PASSED -eq $TOTAL_CHECKS ]; then
    echo "Would you like to reboot now to test auto-start? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "🔄 Rebooting system for auto-start test..."
        sleep 2
        sudo reboot
    else
        echo "Auto-start test can be done manually later with: sudo reboot"
    fi
fi

echo ""
echo "Auto-start verification completed at $(date)"

# Exit with appropriate code
if [ $CHECKS_PASSED -eq $TOTAL_CHECKS ]; then
    exit 0
else
    exit 1
fi
