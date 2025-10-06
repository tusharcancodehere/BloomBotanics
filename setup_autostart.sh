#!/bin/bash
"""
BloomBotanics Complete Automation Setup
Installs everything and enables auto-start on boot
"""

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     🌱 BloomBotanics Auto-Setup & Automation     ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"

# Get current directory
PROJECT_DIR=$(pwd)
USER=$(whoami)

# Step 1: Install dependencies
echo -e "${BLUE}📦 Step 1: Installing system dependencies...${NC}"
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev i2c-tools libgpiod2 python3-smbus

# Enable I2C
sudo raspi-config nonint do_i2c 0
echo -e "${GREEN}✅ System dependencies installed${NC}"

# Step 2: Install Python packages
echo -e "${BLUE}📦 Step 2: Installing Python packages...${NC}"
pip3 install -r requirements.txt
echo -e "${GREEN}✅ Python packages installed${NC}"

# Step 3: Create data directories
echo -e "${BLUE}📁 Step 3: Creating data directories...${NC}"
mkdir -p data/{sensor_data,images,detections,logs,backups}
echo -e "${GREEN}✅ Directories created${NC}"

# Step 4: Update service file with correct paths
echo -e "${BLUE}⚙️  Step 4: Configuring systemd service...${NC}"

# Create updated service file
cat > bloom-botanics.service << EOF
[Unit]
Description=BloomBotanics Agricultural Monitoring System
After=network-online.target multi-user.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin
Environment=PYTHONPATH=$PROJECT_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 $PROJECT_DIR/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Copy service file
sudo cp bloom-botanics.service /etc/systemd/system/
echo -e "${GREEN}✅ Service file installed${NC}"

# Step 5: Enable and start service
echo -e "${BLUE}🚀 Step 5: Enabling auto-start...${NC}"
sudo systemctl daemon-reload
sudo systemctl enable bloom-botanics.service
echo -e "${GREEN}✅ Auto-start enabled${NC}"

# Step 6: Create management scripts
echo -e "${BLUE}🔧 Step 6: Creating management scripts...${NC}"

# Start script
cat > start.sh << 'EOF'
#!/bin/bash
sudo systemctl start bloom-botanics
echo "🌱 BloomBotanics started"
sudo systemctl status bloom-botanics
EOF

# Stop script
cat > stop.sh << 'EOF'
#!/bin/bash
sudo systemctl stop bloom-botanics
echo "🛑 BloomBotanics stopped"
EOF

# Restart script
cat > restart.sh << 'EOF'
#!/bin/bash
sudo systemctl restart bloom-botanics
echo "🔄 BloomBotanics restarted"
sudo systemctl status bloom-botanics
EOF

# Status script
cat > status.sh << 'EOF'
#!/bin/bash
sudo systemctl status bloom-botanics
EOF

# View logs script
cat > logs.sh << 'EOF'
#!/bin/bash
sudo journalctl -u bloom-botanics -f
EOF

chmod +x start.sh stop.sh restart.sh status.sh logs.sh

echo -e "${GREEN}✅ Management scripts created${NC}"

# Final summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           ✅ Setup Complete! 🎉                    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}🌱 BloomBotanics is now configured to:${NC}"
echo -e "   ✅ Auto-start on boot"
echo -e "   ✅ Auto-restart on failure"
echo -e "   ✅ Run continuously in background"
echo ""
echo -e "${YELLOW}📝 Management Commands:${NC}"
echo -e "   ${BLUE}./start.sh${NC}    - Start the system"
echo -e "   ${BLUE}./stop.sh${NC}     - Stop the system"
echo -e "   ${BLUE}./restart.sh${NC}  - Restart the system"
echo -e "   ${BLUE}./status.sh${NC}   - Check system status"
echo -e "   ${BLUE}./logs.sh${NC}     - View live logs"
echo ""
echo -e "${YELLOW}🚀 To start now:${NC}"
echo -e "   ${BLUE}sudo systemctl start bloom-botanics${NC}"
echo ""
echo -e "${YELLOW}🔄 The system will automatically start on next boot!${NC}"
