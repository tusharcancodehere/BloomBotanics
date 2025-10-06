#!/bin/bash
"""
Setup complete logging system for BloomBotanics
Logs to both systemd journal AND project logs folder
"""

PROJECT_DIR=$(pwd)
LOG_DIR="$PROJECT_DIR/data/logs"
USER=$(whoami)

echo "🌱 Setting up BloomBotanics Logging System..."

# Create logs directory
mkdir -p "$LOG_DIR"

# Create log export script
cat > scripts/export_logs.sh << 'EOF'
#!/bin/bash
PROJECT_DIR="/home/pi/BloomBotanics"
LOG_DIR="$PROJECT_DIR/data/logs"
SERVICE_NAME="bloom-botanics"

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Export recent logs
sudo journalctl -u $SERVICE_NAME -n 1000 > "$LOG_DIR/recent.log"

# Export today's logs
sudo journalctl -u $SERVICE_NAME --since "today" > "$LOG_DIR/today.log"

echo "✅ Logs exported to $LOG_DIR"
EOF

chmod +x scripts/export_logs.sh

# Create hourly auto-export script
cat > scripts/auto_log_export.sh << 'EOF'
#!/bin/bash
PROJECT_DIR="/home/pi/BloomBotanics"
LOG_DIR="$PROJECT_DIR/data/logs"
SERVICE_NAME="bloom-botanics"

mkdir -p "$LOG_DIR"
DATE=$(date +%Y-%m-%d)

# Append last hour's logs
sudo journalctl -u $SERVICE_NAME --since "1 hour ago" >> "$LOG_DIR/hourly_${DATE}.log"

# Update recent log
sudo journalctl -u $SERVICE_NAME -n 2000 > "$LOG_DIR/recent.log"

# Cleanup old logs (7 days)
find "$LOG_DIR" -name "hourly_*.log" -mtime +7 -delete
EOF

chmod +x scripts/auto_log_export.sh

# Add to crontab
(crontab -l 2>/dev/null; echo "0 * * * * $PROJECT_DIR/scripts/auto_log_export.sh") | crontab -

# Create convenience scripts
cat > view_logs.sh << 'EOF'
#!/bin/bash
echo "📋 Recent BloomBotanics Logs:"
cat data/logs/recent.log
EOF

cat > view_live_logs.sh << 'EOF'
#!/bin/bash
echo "📡 Live BloomBotanics Logs (Ctrl+C to exit):"
sudo journalctl -u bloom-botanics -f
EOF

chmod +x view_logs.sh view_live_logs.sh

echo "✅ Logging system setup complete!"
echo ""
echo "📝 Log Files Location:"
echo "   $LOG_DIR/recent.log       - Last 1000 lines"
echo "   $LOG_DIR/today.log        - Today's logs"
echo "   $LOG_DIR/hourly_*.log     - Hourly archives"
echo ""
echo "📝 Commands:"
echo "   ./view_logs.sh            - View recent logs"
echo "   ./view_live_logs.sh       - View live logs"
echo "   ./scripts/export_logs.sh  - Export logs manually"
