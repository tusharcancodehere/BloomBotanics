#!/bin/bash
"""
BloomBotanics System Update Script
Updates system packages, Python packages, and BloomBotanics code
"""

set -e

# Configuration
PROJECT_DIR="/home/pi/BloomBotanics"
BACKUP_DIR="$PROJECT_DIR/data/backups"
LOG_FILE="$PROJECT_DIR/data/logs/update_$(date +%Y%m%d_%H%M%S).log"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[UPDATE]${NC} $1" | tee -a "$LOG_FILE"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"; }
print_header() { echo -e "${BLUE}[STAGE]${NC} $1" | tee -a "$LOG_FILE"; }

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

echo "🔄 BloomBotanics System Update" | tee "$LOG_FILE"
echo "==============================" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Function to check if service is running
is_service_running() {
    systemctl is-active --quiet "$1" 2>/dev/null
}

# Function to stop service safely
stop_service() {
    local service="$1"
    if is_service_running "$service"; then
        print_status "Stopping $service service..."
        sudo systemctl stop "$service"
        
        # Wait for service to stop
        for i in {1..10}; do
            if ! is_service_running "$service"; then
                print_status "$service stopped successfully"
                return 0
            fi
            sleep 1
        done
        
        print_warning "$service did not stop gracefully, forcing..."
        sudo systemctl kill "$service" 2>/dev/null || true
        sleep 2
    fi
}

# Function to start service
start_service() {
    local service="$1"
    print_status "Starting $service service..."
    sudo systemctl start "$service"
    
    # Wait for service to start
    for i in {1..15}; do
        if is_service_running "$service"; then
            print_status "$service started successfully"
            return 0
        fi
        sleep 1
    done
    
    print_error "$service failed to start"
    return 1
}

# Pre-update checks
print_header "Pre-Update System Checks"

# Check available disk space
available_space=$(df / | awk 'NR==2 {print $4}')
required_space=1048576  # 1GB in KB

if [ "$available_space" -lt "$required_space" ]; then
    print_error "Insufficient disk space. Available: ${available_space}KB, Required: ${required_space}KB"
    exit 1
fi

print_status "Disk space check passed: ${available_space}KB available"

# Check internet connectivity
if ! ping -c 1 8.8.8.8 >/dev/null 2>&1; then
    print_error "No internet connectivity. Cannot proceed with update."
    exit 1
fi

print_status "Internet connectivity check passed"

# Create pre-update backup
print_header "Creating Pre-Update Backup"
if [ -f "$PROJECT_DIR/scripts/backup_data.sh" ]; then
    print_status "Running backup script..."
    bash "$PROJECT_DIR/scripts/backup_data.sh" >> "$LOG_FILE" 2>&1
    print_status "Pre-update backup completed"
else
    print_warning "Backup script not found, creating simple backup..."
    backup_name="pre_update_backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR/$backup_name"
    
    # Backup critical files
    cp -r "$PROJECT_DIR/data" "$BACKUP_DIR/$backup_name/" 2>/dev/null || true
    cp "$PROJECT_DIR/config.py" "$BACKUP_DIR/$backup_name/" 2>/dev/null || true
    cp "$PROJECT_DIR/main.py" "$BACKUP_DIR/$backup_name/" 2>/dev/null || true
    
    print_status "Simple backup created: $backup_name"
fi

# Stop BloomBotanics service
print_header "Service Management"
stop_service "bloom-botanics"

# System package updates
print_header "System Package Updates"
print_status "Updating package lists..."
sudo apt update >> "$LOG_FILE" 2>&1

print_status "Upgrading system packages..."
sudo apt upgrade -y >> "$LOG_FILE" 2>&1

print_status "Installing any missing dependencies..."
sudo apt install -y python3-pip python3-dev python3-venv git i2c-tools \
    python3-rpi.gpio python3-serial python3-spidev python3-picamera2 \
    python3-opencv build-essential cmake >> "$LOG_FILE" 2>&1

# Python package updates
print_header "Python Package Updates"
print_status "Updating pip..."
python3 -m pip install --upgrade pip --break-system-packages >> "$LOG_FILE" 2>&1

if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    print_status "Updating Python packages from requirements.txt..."
    pip install --upgrade --break-system-packages -r "$PROJECT_DIR/requirements.txt" >> "$LOG_FILE" 2>&1
else
    print_warning "requirements.txt not found, updating common packages..."
    pip install --upgrade --break-system-packages \
        RPi.GPIO adafruit-circuitpython-dht picamera2 ultralytics \
        opencv-python pyserial RPLCD spidev numpy psutil >> "$LOG_FILE" 2>&1
fi

# BloomBotanics code update (if using git)
print_header "BloomBotanics Code Update"
cd "$PROJECT_DIR"

if [ -d ".git" ]; then
    print_status "Updating BloomBotanics code from repository..."
    
    # Stash local changes
    git stash push -m "Pre-update stash $(date)" >> "$LOG_FILE" 2>&1 || true
    
    # Pull latest changes
    git pull origin main >> "$LOG_FILE" 2>&1
    
    # Apply stashed changes if any
    if git stash list | grep -q "Pre-update stash"; then
        print_warning "Local changes were stashed. Review with 'git stash list'"
    fi
    
    print_status "Code update completed"
else
    print_warning "Not a git repository. Manual code update required."
fi

# Update file permissions
print_header "File Permissions Update"
print_status "Updating file permissions..."
chmod +x "$PROJECT_DIR"/*.sh "$PROJECT_DIR"/main.py 2>/dev/null || true
chmod +x "$PROJECT_DIR"/scripts/*.sh 2>/dev/null || true

# Systemd service update
print_header "Service Configuration Update"
if [ -f "$PROJECT_DIR/services/bloom-botanics.service" ]; then
    print_status "Updating systemd service configuration..."
    
    # Check if service file has changed
    if ! diff -q "$PROJECT_DIR/services/bloom-botanics.service" "/etc/systemd/system/bloom-botanics.service" >/dev/null 2>&1; then
        print_status "Service file updated, installing new version..."
        sudo cp "$PROJECT_DIR/services/bloom-botanics.service" /etc/systemd/system/
        sudo systemctl daemon-reload
        print_status "Service configuration updated"
    else
        print_status "Service configuration unchanged"
    fi
fi

# Hardware interface verification
print_header "Hardware Interface Verification"
print_status "Verifying hardware interfaces are enabled..."

# Check and enable interfaces if needed
interfaces_changed=false

if ! raspi-config nonint get_camera | grep -q "0"; then
    print_status "Enabling camera interface..."
    sudo raspi-config nonint do_camera 0
    interfaces_changed=true
fi

if ! raspi-config nonint get_i2c | grep -q "0"; then
    print_status "Enabling I2C interface..."
    sudo raspi-config nonint do_i2c 0
    interfaces_changed=true
fi

if ! raspi-config nonint get_spi | grep -q "0"; then
    print_status "Enabling SPI interface..."
    sudo raspi-config nonint do_spi 0
    interfaces_changed=true
fi

if [ "$interfaces_changed" = true ]; then
    print_warning "Hardware interfaces were modified. Reboot recommended after update."
fi

# System cleanup
print_header "System Cleanup"
print_status "Cleaning up package cache..."
sudo apt autoremove -y >> "$LOG_FILE" 2>&1
sudo apt autoclean >> "$LOG_FILE" 2>&1

print_status "Cleaning up Python cache..."
find "$PROJECT_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$PROJECT_DIR" -name "*.pyc" -delete 2>/dev/null || true

# Log rotation cleanup
print_status "Rotating old logs..."
find "$PROJECT_DIR/data/logs" -name "*.log" -mtime +30 -delete 2>/dev/null || true
find "$PROJECT_DIR/data/logs" -name "health_report_*.json" -mtime +7 -delete 2>/dev/null || true

# Configuration validation
print_header "Configuration Validation"
if [ -f "$PROJECT_DIR/config.py" ]; then
    print_status "Validating configuration..."
    
    # Check for placeholder phone number
    if grep -q "+919876543210" "$PROJECT_DIR/config.py"; then
        print_warning "⚠️ Please update PHONE_NUMBER in config.py"
    fi
    
    # Validate Python syntax
    if python3 -m py_compile "$PROJECT_DIR/config.py" 2>/dev/null; then
        print_status "Configuration syntax is valid"
    else
        print_error "Configuration syntax error detected!"
        print_error "Please check config.py before starting service"
    fi
fi

# Quick system test
print_header "Post-Update Testing"
if [ -f "$PROJECT_DIR/quick_test.py" ]; then
    print_status "Running quick system test..."
    if python3 "$PROJECT_DIR/quick_test.py" >> "$LOG_FILE" 2>&1; then
        print_status "System test passed"
    else
        print_warning "System test had warnings. Check log for details."
    fi
fi

# Start service
print_header "Service Restart"
if start_service "bloom-botanics"; then
    print_status "BloomBotanics service started successfully"
    
    # Wait a moment and check if service is stable
    sleep 5
    if is_service_running "bloom-botanics"; then
        print_status "Service is running stably"
    else
        print_error "Service stopped unexpectedly. Check logs:"
        print_error "sudo journalctl -u bloom-botanics -n 20"
    fi
else
    print_error "Failed to start BloomBotanics service"
    print_error "Check service status: sudo systemctl status bloom-botanics"
    print_error "View logs: sudo journalctl -u bloom-botanics -f"
fi

# Generate post-update report
print_header "Post-Update Report"
echo "" | tee -a "$LOG_FILE"
echo "📊 Update Summary" | tee -a "$LOG_FILE"
echo "=================" | tee -a "$LOG_FILE"
echo "Completed: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# System info
echo "System Information:" | tee -a "$LOG_FILE"
echo "  OS Version: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)" | tee -a "$LOG_FILE"
echo "  Kernel: $(uname -r)" | tee -a "$LOG_FILE"
echo "  Python: $(python3 --version)" | tee -a "$LOG_FILE"
echo "  Uptime: $(uptime -p)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Service status
echo "Service Status:" | tee -a "$LOG_FILE"
if is_service_running "bloom-botanics"; then
    echo "  ✅ BloomBotanics: Running" | tee -a "$LOG_FILE"
else
    echo "  ❌ BloomBotanics: Not Running" | tee -a "$LOG_FILE"
fi

if is_service_running "rpi-connect"; then
    echo "  ✅ Pi Connect: Running" | tee -a "$LOG_FILE"
else
    echo "  ⚠️ Pi Connect: Not Running" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"

# Disk usage after update
echo "Disk Usage:" | tee -a "$LOG_FILE"
df -h / | tail -1 | awk '{print "  Total: " $2 ", Used: " $3 ", Available: " $4 ", Usage: " $5}' | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Recommendations
echo "📋 Post-Update Recommendations:" | tee -a "$LOG_FILE"
echo "  1. Monitor service for 24 hours: sudo journalctl -u bloom-botanics -f" | tee -a "$LOG_FILE"
echo "  2. Test SMS alerts and irrigation" | tee -a "$LOG_FILE"
echo "  3. Verify Pi Connect access: connect.raspberrypi.com" | tee -a "$LOG_FILE"
echo "  4. Check system health: ./scripts/system_health.py" | tee -a "$LOG_FILE"

if [ "$interfaces_changed" = true ]; then
    echo "  5. REBOOT REQUIRED: sudo reboot (hardware interfaces were changed)" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"

print_status "🎉 BloomBotanics system update completed successfully!"
print_status "📄 Full log available at: $LOG_FILE"

if [ "$interfaces_changed" = true ]; then
    print_warning "🔄 REBOOT REQUIRED - Hardware interfaces were modified"
    echo ""
    echo "Reboot now? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_status "Rebooting system..."
        sudo reboot
    fi
fi

echo ""
echo "Update completed at $(date)"
