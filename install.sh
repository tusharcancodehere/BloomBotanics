#!/bin/bash
set -e

echo "🌱 Installing BloomBotanics Agricultural Monitoring System"
echo "========================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_header() { echo -e "${BLUE}[STAGE]${NC} $1"; }

# Verify we're on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    print_error "This script must be run on a Raspberry Pi!"
    exit 1
fi

print_header "System Update and Dependencies"
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

print_status "Installing system dependencies..."
sudo apt install -y \
    python3-pip python3-dev python3-venv git i2c-tools \
    python3-rpi.gpio python3-serial python3-spidev \
    python3-picamera2 python3-opencv \
    build-essential cmake

print_header "Hardware Interface Configuration"
print_status "Enabling hardware interfaces..."
sudo raspi-config nonint do_camera 0      # Enable camera
sudo raspi-config nonint do_i2c 0         # Enable I2C
sudo raspi-config nonint do_spi 0         # Enable SPI
sudo raspi-config nonint do_serial 1      # Enable serial (disable console)

print_header "Python Environment Setup"
print_status "Installing Python packages..."
pip install --break-system-packages -r requirements.txt

print_header "Project Configuration"
print_status "Creating data directories..."
mkdir -p data/{sensor_data,images,detections,logs,backups}
mkdir -p scripts logs

print_status "Setting permissions..."
chmod +x main.py deploy.sh test_autostart.sh
chmod +x scripts/*.sh 2>/dev/null || true

print_header "System Service Installation"
print_status "Installing systemd service..."
sudo cp services/bloom-botanics.service /etc/systemd/system/
sudo chmod 644 /etc/systemd/system/bloom-botanics.service
sudo systemctl daemon-reload
sudo systemctl enable bloom-botanics.service

print_status "Configuring log rotation..."
sudo tee /etc/logrotate.d/bloom-botanics > /dev/null <<EOF
/home/pi/BloomBotanics/data/logs/*.log {
    daily
    rotate 30
    compress
    missingok
    notifempty
    create 644 pi pi
    postrotate
        systemctl reload bloom-botanics || true
    endscript
}
EOF

print_header "Pi Connect Configuration"
print_status "Ensuring Pi Connect is available..."
if ! command -v rpi-connect &> /dev/null; then
    sudo apt install -y rpi-connect
fi

print_status "Configuring Pi Connect for persistent connection..."
sudo systemctl enable rpi-connect 2>/dev/null || true
loginctl enable-linger pi 2>/dev/null || true

print_header "System Health Monitoring"
print_status "Setting up system health monitoring..."
cat > scripts/health_check.sh << 'EOF'
#!/bin/bash
# BloomBotanics Health Check Script
echo "🏥 BloomBotanics System Health Check"
echo "=================================="
echo "Service Status:"
sudo systemctl status bloom-botanics --no-pager -l
echo ""
echo "Recent Logs (last 10 lines):"
sudo journalctl -u bloom-botanics -n 10 --no-pager
echo ""
echo "System Resources:"
df -h / | tail -1
free -h | head -2
cat /sys/class/thermal/thermal_zone0/temp | awk '{print "CPU Temp: " $1/1000 "°C"}'
EOF

chmod +x scripts/health_check.sh

print_header "Verification and Testing"
# Create verification script
cat > test_autostart.sh << 'EOF'
#!/bin/bash
echo "🧪 BloomBotanics Auto-Start Verification"
echo "======================================="

# Test service file
if [ -f "/etc/systemd/system/bloom-botanics.service" ]; then
    echo "✅ Service file exists"
else
    echo "❌ Service file missing"
    exit 1
fi

# Test service enabled
if systemctl is-enabled bloom-botanics.service | grep -q "enabled"; then
    echo "✅ Service enabled for auto-start"
else
    echo "❌ Service NOT enabled"
    exit 1
fi

# Test service status
STATUS=$(systemctl is-active bloom-botanics.service 2>/dev/null || echo "inactive")
echo "📊 Service Status: $STATUS"

echo ""
echo "Configuration Check:"
if [ -f "config.py" ]; then
    PHONE=$(grep "PHONE_NUMBER" config.py | head -1)
    echo "📱 $PHONE"
    if [[ "$PHONE" == *"+919876543210"* ]]; then
        echo "⚠️  WARNING: Please update PHONE_NUMBER in config.py"
    fi
fi

echo ""
echo "🔄 To test auto-start: sudo reboot"
echo "📱 After reboot: connect.raspberrypi.com"
EOF

chmod +x test_autostart.sh

print_header "Final Configuration"
print_status "Installation completed successfully!"

echo ""
echo "🎉 BloomBotanics Installation Complete!"
echo "======================================"
echo ""
echo "📋 Next Steps:"
echo "  1. Update PHONE_NUMBER in config.py:"
echo "     nano config.py"
echo ""
echo "  2. Test the installation:"
echo "     ./test_autostart.sh"
echo ""
echo "  3. Run system manually (for testing):"
echo "     python3 main.py"
echo ""
echo "  4. Start service:"
echo "     sudo systemctl start bloom-botanics"
echo ""
echo "  5. Check status:"
echo "     sudo systemctl status bloom-botanics"
echo ""
echo "  6. View logs:"
echo "     sudo journalctl -u bloom-botanics -f"
echo ""
echo "🌐 Remote Access:"
echo "  Pi Connect: https://connect.raspberrypi.com"
echo ""
echo "🔧 System Commands:"
echo "  Health check: ./scripts/health_check.sh"
echo "  Stop service: sudo systemctl stop bloom-botanics"
echo "  Disable auto-start: sudo systemctl disable bloom-botanics"
echo ""
print_warning "IMPORTANT: Reboot system to test auto-start: sudo reboot"
echo ""
