# BloomBotanics Installation Guide

## Quick Install (Recommended)

curl -sSL https://github.com/yourusername/BloomBotanics/install.sh | bash

## Manual Installation Steps

### 1. Prepare Raspberry Pi
# Update system
sudo apt update && sudo apt upgrade -y

# Enable interfaces  
sudo raspi-config
# Interface Options → Enable: SSH, Camera, I2C, Serial (hardware only)

### 2. Download BloomBotanics
cd /home/pi
git clone https://github.com/yourusername/BloomBotanics.git
cd BloomBotanics
chmod +x *.sh

### 3. Install Dependencies
# System packages
sudo apt install -y python3-pip python3-dev python3-rpi.gpio \
    python3-serial python3-picamera2 python3-opencv i2c-tools

# Python packages
pip install --break-system-packages -r requirements.txt

### 4. Configure System
# Edit configuration
nano config.py

# REQUIRED: Update your phone number
PHONE_NUMBER = "+919876543210"  # Change this!

### 5. Install Service
# Install auto-start service
sudo cp services/bloom-botanics.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable bloom-botanics.service

### 6. Test Installation
# Test hardware
python3 scripts/test_sensors.py

# Test auto-start setup
./test_autostart.sh

# Start service
sudo systemctl start bloom-botanics

## Pi Connect Setup

# Install Pi Connect (for remote access)
sudo apt install -y rpi-connect
sudo systemctl enable rpi-connect

# Sign up at: https://id.raspberrypi.com
# Your Pi will appear at: https://connect.raspberrypi.com

## Final Testing

# Check service status
sudo systemctl status bloom-botanics

# View live logs
sudo journalctl -u bloom-botanics -f

# Test reboot (auto-start test)
sudo reboot

## Common Issues

Import Errors:
pip install --break-system-packages -r requirements.txt

GPIO Errors:
sudo usermod -a -G gpio pi
sudo reboot

Service Won't Start:
sudo journalctl -u bloom-botanics -n 20
python3 main.py  # Run manually for debugging

That's it! Your BloomBotanics system is now installed and will start automatically on every boot!
