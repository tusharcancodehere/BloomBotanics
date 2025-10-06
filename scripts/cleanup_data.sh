#!/bin/bash
"""
BloomBotanics Data Cleanup Script
Cleans up old data files, logs, and temporary files
"""

set -e

PROJECT_DIR="/home/pi/BloomBotanics"
DATA_DIR="$PROJECT_DIR/data"

# Default retention periods (days)
LOG_RETENTION=30
IMAGE_RETENTION=90
DETECTION_RETENTION=60
SENSOR_DATA_RETENTION=365

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[CLEANUP]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "🧹 BloomBotanics Data Cleanup"
echo "============================="
echo "Started: $(date)"
echo ""

# Function to cleanup directory by age
cleanup_by_age() {
    local directory="$1"
    local days="$2"
    local description="$3"
    
    if [ ! -d "$directory" ]; then
        print_warning "Directory not found: $directory"
        return
    fi
    
    print_status "$description (older than $days days)..."
    
    # Count files before cleanup
    before_count=$(find "$directory" -type f | wc -l)
    before_size=$(du -sh "$directory" 2>/dev/null | cut -f1 || echo "0B")
    
    # Remove old files
    deleted_count=0
    while IFS= read -r -d '' file; do
        rm "$file" 2>/dev/null && ((deleted_count++))
    done < <(find "$directory" -type f -mtime +$days -print0 2>/dev/null)
    
    # Count files after cleanup
    after_count=$(find "$directory" -type f | wc -l)
    after_size=$(du -sh "$directory" 2>/dev/null | cut -f1 || echo "0B")
    
    print_status "  Deleted: $deleted_count files"
    print_status "  Before: $before_count files ($before_size)"
    print_status "  After: $after_count files ($after_size)"
}

# Function to cleanup large files
cleanup_large_files() {
    local directory="$1"
    local max_size_mb="$2"
    local description="$3"
    
    if [ ! -d "$directory" ]; then
        return
    fi
    
    print_status "$description (larger than ${max_size_mb}MB)..."
    
    # Find and remove large files
    large_files=$(find "$directory" -type f -size +${max_size_mb}M 2>/dev/null)
    
    if [ -n "$large_files" ]; then
        echo "$large_files" | while read -r file; do
            file_size=$(du -h "$file" | cut -f1)
            print_warning "Removing large file: $(basename "$file") ($file_size)"
            rm "$file" 2>/dev/null || print_error "Failed to remove: $file"
        done
    else
        print_status "  No large files found"
    fi
}

# Function to compress old files
compress_old_files() {
    local directory="$1"
    local days="$2"
    local description="$3"
    
    if [ ! -d "$directory" ]; then
        return
    fi
    
    print_status "$description (older than $days days)..."
    
    # Find old uncompressed files
    find "$directory" -type f -name "*.log" -mtime +$days 2>/dev/null | while read -r file; do
        if [[ ! "$file" =~ \.gz$ ]]; then
            print_status "Compressing: $(basename "$file")"
            gzip "$file" 2>/dev/null || print_error "Failed to compress: $file"
        fi
    done
    
    find "$directory" -type f -name "*.json" -mtime +$days 2>/dev/null | while read -r file; do
        if [[ ! "$file" =~ \.gz$ ]]; then
            print_status "Compressing: $(basename "$file")"
            gzip "$file" 2>/dev/null || print_error "Failed to compress: $file"
        fi
    done
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --logs)
            LOG_RETENTION="$2"
            shift 2
            ;;
        --images)
            IMAGE_RETENTION="$2"
            shift 2
            ;;
        --detections)
            DETECTION_RETENTION="$2"
            shift 2
            ;;
        --sensor-data)
            SENSOR_DATA_RETENTION="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            echo "Usage: $0 [--logs DAYS] [--images DAYS] [--detections DAYS] [--sensor-data DAYS] [--dry-run]"
            exit 1
            ;;
    esac
done

if [ "${DRY_RUN:-0}" -eq 1 ]; then
    print_warning "DRY RUN MODE - No files will actually be deleted"
    echo ""
fi

# Display cleanup plan
echo "Cleanup Plan:"
echo "  Log files: older than $LOG_RETENTION days"
echo "  Images: older than $IMAGE_RETENTION days"
echo "  Detections: older than $DETECTION_RETENTION days"
echo "  Sensor data: older than $SENSOR_DATA_RETENTION days"
echo ""

# Cleanup operations
if [ "${DRY_RUN:-0}" -eq 0 ]; then
    # Clean up logs
    cleanup_by_age "$DATA_DIR/logs" "$LOG_RETENTION" "Cleaning up log files"
    
    # Clean up images
    cleanup_by_age "$DATA_DIR/images" "$IMAGE_RETENTION" "Cleaning up crop images"
    
    # Clean up detection images
    cleanup_by_age "$DATA_DIR/detections" "$DETECTION_RETENTION" "Cleaning up detection images"
    
    # Clean up sensor data (be more conservative)
    cleanup_by_age "$DATA_DIR/sensor_data" "$SENSOR_DATA_RETENTION" "Cleaning up old sensor data"
    
    # Clean up backup files (keep recent backups only)
    cleanup_by_age "$DATA_DIR/backups" "14" "Cleaning up old backups"
    
    # Clean up system cache and temporary files
    print_status "Cleaning up system cache..."
    find "$PROJECT_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$PROJECT_DIR" -name "*.pyc" -delete 2>/dev/null || true
    find "$PROJECT_DIR" -name "*.tmp" -delete 2>/dev/null || true
    
    # Clean up large files (over 100MB)
    cleanup_large_files "$DATA_DIR" "100" "Cleaning up large files"
    
    # Compress old files
    compress_old_files "$DATA_DIR/logs" "7" "Compressing old log files"
    compress_old_files "$DATA_DIR/sensor_data" "30" "Compressing old sensor data"
    
    # Clean up journal logs
    print_status "Cleaning up system journal..."
    sudo journalctl --vacuum-time=30d >/dev/null 2>&1 || true
    sudo journalctl --vacuum-size=100M >/dev/null 2>&1 || true
    
else
    # Dry run mode - just show what would be deleted
    echo "DRY RUN - Files that would be deleted:"
    
    find "$DATA_DIR/logs" -type f -mtime +$LOG_RETENTION 2>/dev/null | head -10 | while read -r file; do
        echo "  LOG: $(basename "$file")"
    done
    
    find "$DATA_DIR/images" -type f -mtime +$IMAGE_RETENTION 2>/dev/null | head -10 | while read -r file; do
        echo "  IMAGE: $(basename "$file")"
    done
    
    find "$DATA_DIR/detections" -type f -mtime +$DETECTION_RETENTION 2>/dev/null | head -10 | while read -r file; do
        echo "  DETECTION: $(basename "$file")"
    done
    
    find "$DATA_DIR/sensor_data" -type f -mtime +$SENSOR_DATA_RETENTION 2>/dev/null | head -10 | while read -r file; do
        echo "  SENSOR: $(basename "$file")"
    done
fi

# Final disk usage report
echo ""
echo "📊 Disk Usage After Cleanup:"
echo "============================="

if [ -d "$DATA_DIR" ]; then
    echo "Data Directory Sizes:"
    du -h "$DATA_DIR"/* 2>/dev/null | sort -hr | while read -r size dir; do
        echo "  $(basename "$dir"): $size"
    done
fi

echo ""
echo "Total Project Size: $(du -sh "$PROJECT_DIR" | cut -f1)"
echo "Available Space: $(df -h / | tail -1 | awk '{print $4}')"
echo ""

print_status "🎉 Cleanup completed at $(date)"

# Suggestions
echo "💡 Suggestions:"
echo "  - Run this script weekly: crontab -e"
echo "  - Add line: 0 2 * * 0 /home/pi/BloomBotanics/scripts/cleanup_data.sh"
echo "  - Monitor disk usage: df -h"
echo "  - Backup important data before cleanup"
