#!/bin/bash
"""
BloomBotanics Auto Setup & Validation Script
Automatically installs, configures, and validates the entire system
"""

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     🌱 BloomBotanics Auto Setup & Validator      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Warning: Not running on Raspberry Pi${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo -e "${BLUE}📦 Updating system packages...${NC}"
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Python and dependencies
echo -e "${BLUE}🐍 Installing Python dependencies...${NC}"
sudo apt-get install -y python3 python3-pip python3-venv git

# Install system libraries
echo -e "${BLUE}📚 Installing system libraries...${NC}"
sudo apt-get install -y \
    i2c-tools \
    libgpiod2 \
    python3-dev \
    python3-smbus \
    libatlas-base-dev \
    libjpeg-dev \
    libopenjp2-7 \
    libtiff5

# Enable I2C and Camera
echo -e "${BLUE}⚙️  Enabling I2C and Camera interfaces...${NC}"
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_camera 0

# Create virtual environment
if [ ! -d "venv" ]; then
    echo -e "${BLUE}🔧 Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install Python packages
echo -e "${BLUE}📥 Installing Python packages...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo -e "${BLUE}📁 Creating project directories...${NC}"
mkdir -p data/{sensor_data,images,detections,logs,backups}

# Set up systemd service for auto-start
echo -e "${BLUE}🚀 Setting up auto-start service...${NC}"
sudo tee /etc/systemd/system/bloombotanics.service > /dev/null << EOF
[Unit]
Description=BloomBotanics Agricultural Monitoring System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/venv/bin"
ExecStart=$(pwd)/venv/bin/python3 $(pwd)/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable bloombotanics.service

echo -e "${GREEN}✅ Installation complete!${NC}"
echo -e "${BLUE}Running system validation...${NC}"

# Run validation
python3 validate_system.py

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           Setup Complete! 🎉                       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "To start the system manually:"
echo -e "  ${YELLOW}python3 main.py${NC}"
echo ""
echo -e "To enable auto-start on boot:"
echo -e "  ${YELLOW}sudo systemctl start bloombotanics${NC}"
echo ""
echo -e "To view logs:"
echo -e "  ${YELLOW}sudo journalctl -u bloombotanics -f${NC}"
