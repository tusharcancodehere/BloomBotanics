#!/bin/bash
set -e

# BloomBotanics Deployment Script - Laptop to Raspberry Pi
# Usage: ./deploy.sh [PI_IP] [USERNAME]

PI_IP=${1:-"raspberrypi.local"}
USERNAME=${2:-"pi"}

echo "🚀 Deploying BloomBotanics to Raspberry Pi"
echo "Target: $USERNAME@$PI_IP"
echo "========================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[DEPLOY]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_header() { echo -e "${BLUE}[STAGE]${NC} $1"; }

# Verify local files exist
if [ ! -f "main.py" ] || [ ! -f "config.py" ]; then
    print_error "main.py or config.py not found in current directory"
    exit 1
fi

print_header "Connection Test"
print_status "Testing SSH connection to $PI_IP..."
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes $USERNAME@$PI_IP exit 2>/dev/null; then
    print_error "Cannot connect to $USERNAME@$PI_IP"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check Pi is powered on and connected to network"
    echo "  2. Verify SSH is enabled: sudo systemctl enable ssh"
    echo "  3. Try IP address instead of hostname"
    echo "  4. Check firewall settings"
    echo ""
    exit 1
fi
print_status "✅ SSH connection successful"

print_header "Pre-Deployment Backup"
print_status "Creating backup of existing installation..."
ssh $USERNAME@$PI_IP "
    if [ -d BloomBotanics ]; then
        backup_name=BloomBotanics_backup_\$(date +%Y%m%d_%H%M%S)
        cp -r BloomBotanics \$backup_name
        echo 'Backup created: \$backup_name'
    fi
"

print_status "Stopping existing service..."
ssh $USERNAME@$PI_IP "sudo systemctl stop bloom-botanics 2>/dev/null || true"

print_header "File Transfer"
print_status "Transferring project files..."
rsync -avz --progress \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='data/' \
    --exclude='*.log' \
    --exclude='.env' \
    ./ $USERNAME@$PI_IP:~/BloomBotanics/

print_status "✅ Files transferred successfully"

print_header "Remote Installation"
print_status "Setting permissions..."
ssh $USERNAME@$PI_IP "
    cd BloomBotanics
    chmod +x *.sh main.py
    chmod +x scripts/*.sh 2>/dev/null || true
"

print_status "Running installation on Pi..."
ssh $USERNAME@$PI_IP "cd BloomBotanics && ./install.sh"

print_header "Deployment Verification"
print_status "Testing auto-start configuration..."
ssh $USERNAME@$PI_IP "cd BloomBotanics && ./test_autostart.sh"

print_status "Starting service..."
ssh $USERNAME@$PI_IP "sudo systemctl start bloom-botanics"

# Wait for service to start
sleep 5

print_status "Checking service status..."
if ssh $USERNAME@$PI_IP "systemctl is-active bloom-botanics" | grep -q "active"; then
    print_status "✅ Service running successfully!"
else
    print_error "❌ Service failed to start"
    print_status "Checking logs..."
    ssh $USERNAME@$PI_IP "sudo journalctl -u bloom-botanics -n 20 --no-pager"
    exit 1
fi

print_header "Deployment Complete"
echo ""
echo "🎉 BloomBotanics Successfully Deployed!"
echo "====================================="
echo ""
echo "📋 System Status:"
echo "  Target: $USERNAME@$PI_IP"
echo "  Service: ✅ Running"
echo "  Auto-start: ✅ Enabled"
echo ""
echo "🔧 Management Commands:"
echo "  View logs: ssh $USERNAME@$PI_IP 'sudo journalctl -u bloom-botanics -f'"
echo "  Stop system: ssh $USERNAME@$PI_IP 'sudo systemctl stop bloom-botanics'"
echo "  Restart: ssh $USERNAME@$PI_IP 'sudo systemctl restart bloom-botanics'"
echo "  Health check: ssh $USERNAME@$PI_IP 'cd BloomBotanics && ./scripts/health_check.sh'"
echo ""
echo "🌐 Remote Access:"
echo "  Pi Connect: https://connect.raspberrypi.com"
echo "  SSH: ssh $USERNAME@$PI_IP"
echo ""
echo "⚙️ Configuration:"
echo "  Edit config: ssh $USERNAME@$PI_IP 'nano BloomBotanics/config.py'"
echo "  Update phone: Change PHONE_NUMBER in config.py"
echo ""
print_warning "Remember to update PHONE_NUMBER in config.py before production use!"
echo ""
