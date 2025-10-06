#!/bin/bash
"""
BloomBotanics Data Backup Script
Automatically backs up sensor data, images, and system logs
"""

set -e

# Configuration
BACKUP_DIR="/home/pi/BloomBotanics/data/backups"
DATA_DIR="/home/pi/BloomBotanics/data"
RETENTION_DAYS=30
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="bloom_backup_${TIMESTAMP}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[BACKUP]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "🗄️ BloomBotanics Data Backup System"
echo "==================================="
echo "Backup Name: $BACKUP_NAME"
echo "Target Directory: $BACKUP_DIR"
echo ""

# Create backup directory
print_status "Creating backup directory..."
mkdir -p "$BACKUP_DIR/$BACKUP_NAME"

# Function to backup directory with progress
backup_directory() {
    local src_dir="$1"
    local dest_name="$2"
    local description="$3"
    
    if [ -d "$src_dir" ]; then
        print_status "$description"
        
        # Count files for progress
        file_count=$(find "$src_dir" -type f | wc -l)
        print_status "Found $file_count files in $src_dir"
        
        # Create tar archive with progress
        tar -czf "$BACKUP_DIR/$BACKUP_NAME/${dest_name}.tar.gz" -C "$(dirname "$src_dir")" "$(basename "$src_dir")" 2>/dev/null
        
        if [ $? -eq 0 ]; then
            archive_size=$(du -h "$BACKUP_DIR/$BACKUP_NAME/${dest_name}.tar.gz" | cut -f1)
            print_status "✅ ${description} completed - Archive size: $archive_size"
        else
            print_error "❌ Failed to backup $description"
        fi
    else
        print_warning "Directory not found: $src_dir"
    fi
}

# Backup sensor data
backup_directory "$DATA_DIR/sensor_data" "sensor_data" "Backing up sensor data..."

# Backup images
backup_directory "$DATA_DIR/images" "images" "Backing up crop images..."

# Backup detection images
backup_directory "$DATA_DIR/detections" "detections" "Backing up threat detection images..."

# Backup logs
backup_directory "$DATA_DIR/logs" "logs" "Backing up system logs..."

# Backup configuration
print_status "Backing up configuration files..."
config_files=(
    "/home/pi/BloomBotanics/config.py"
    "/home/pi/BloomBotanics/main.py"
    "/etc/systemd/system/bloom-botanics.service"
)

config_backup_dir="$BACKUP_DIR/$BACKUP_NAME/config"
mkdir -p "$config_backup_dir"

for file in "${config_files[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$config_backup_dir/"
        print_status "✅ Backed up: $(basename "$file")"
    fi
done

# Create system information snapshot
print_status "Creating system information snapshot..."
info_file="$BACKUP_DIR/$BACKUP_NAME/system_info.txt"

{
    echo "BloomBotanics System Information"
    echo "==============================="
    echo "Backup Date: $(date)"
    echo "System Uptime: $(uptime)"
    echo ""
    echo "Raspberry Pi Information:"
    cat /proc/cpuinfo | grep -E "(Hardware|Revision|Serial)" || echo "N/A"
    echo ""
    echo "Memory Usage:"
    free -h
    echo ""
    echo "Disk Usage:"
    df -h
    echo ""
    echo "Python Packages:"
    pip list | grep -E "(RPi|adafruit|opencv|ultralytics|serial|RPLCD)" || echo "N/A"
    echo ""
    echo "System Services:"
    systemctl is-active bloom-botanics || echo "Service not active"
    echo ""
    echo "Network Configuration:"
    ip addr show | grep -E "(inet |wlan|eth)" || echo "N/A"
} > "$info_file"

print_status "✅ System information saved"

# Calculate total backup size
total_size=$(du -sh "$BACKUP_DIR/$BACKUP_NAME" | cut -f1)
print_status "Total backup size: $total_size"

# Create backup manifest
manifest_file="$BACKUP_DIR/$BACKUP_NAME/MANIFEST.txt"
{
    echo "BloomBotanics Backup Manifest"
    echo "=============================="
    echo "Backup Name: $BACKUP_NAME"
    echo "Created: $(date)"
    echo "Total Size: $total_size"
    echo ""
    echo "Contents:"
    ls -la "$BACKUP_DIR/$BACKUP_NAME/"
    echo ""
    echo "File Counts:"
    find "$BACKUP_DIR/$BACKUP_NAME" -name "*.tar.gz" -exec bash -c 'echo "Archive: $(basename "$1"), Files: $(tar -tzf "$1" | wc -l)"' _ {} \;
} > "$manifest_file"

# Cleanup old backups
print_status "Cleaning up old backups (older than $RETENTION_DAYS days)..."
find "$BACKUP_DIR" -maxdepth 1 -type d -name "bloom_backup_*" -mtime +$RETENTION_DAYS -exec rm -rf {} \; 2>/dev/null || true

old_count=$(find "$BACKUP_DIR" -maxdepth 1 -type d -name "bloom_backup_*" | wc -l)
print_status "Remaining backups: $old_count"

# Create quick restore script
restore_script="$BACKUP_DIR/$BACKUP_NAME/restore.sh"
cat > "$restore_script" << EOF
#!/bin/bash
# BloomBotanics Restore Script
# Auto-generated on $(date)

echo "🔄 BloomBotanics Data Restore"
echo "============================="
echo "Restoring backup: $BACKUP_NAME"
echo ""

# Stop service
sudo systemctl stop bloom-botanics 2>/dev/null || true

# Extract archives
tar -xzf sensor_data.tar.gz -C /home/pi/BloomBotanics/data/
tar -xzf images.tar.gz -C /home/pi/BloomBotanics/data/
tar -xzf detections.tar.gz -C /home/pi/BloomBotanics/data/
tar -xzf logs.tar.gz -C /home/pi/BloomBotanics/data/

# Restore config files
cp config/* /home/pi/BloomBotanics/

echo "✅ Restore completed!"
echo "Start service: sudo systemctl start bloom-botanics"
EOF

chmod +x "$restore_script"

echo ""
echo "🎉 Backup Completed Successfully!"
echo "================================="
echo "Backup Location: $BACKUP_DIR/$BACKUP_NAME"
echo "Total Size: $total_size"
echo "Files Included:"
echo "  - Sensor data archives"
echo "  - Image archives" 
echo "  - System logs"
echo "  - Configuration files"
echo "  - System information snapshot"
echo "  - Restore script"
echo ""
echo "To restore this backup:"
echo "  cd $BACKUP_DIR/$BACKUP_NAME"
echo "  ./restore.sh"
echo ""

# Optional: Upload to cloud storage (if configured)
if [ -f "/home/pi/.config/rclone/rclone.conf" ]; then
    print_status "Cloud backup configuration detected"
    echo "To upload to cloud: rclone copy $BACKUP_DIR/$BACKUP_NAME remote:BloomBotanics/backups/"
fi

print_status "Backup script completed at $(date)"
