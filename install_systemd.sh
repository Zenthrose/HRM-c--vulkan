#!/usr/bin/env bash

# HRM System Installation Script
# Sets up HRM as a systemd service with auto-boot and idle processing

set -euo pipefail

# Cross-platform detection
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "❌ This script is for Linux/macOS systems with systemd."
    echo "For Windows, use install_windows_service.ps1"
    exit 1
fi

HRM_USER="hrm"
HRM_HOME="/opt/hrm"
HRM_SERVICE_NAME="hrm"
SERVICE_FILE="/etc/systemd/system/${HRM_SERVICE_NAME}.service"

echo "🚀 Installing HRM Self-Evolving AI System..."

# Create HRM user if it doesn't exist
if ! id "$HRM_USER" &>/dev/null; then
    echo "Creating HRM user: $HRM_USER"
    sudo useradd -r -s /bin/bash -d "$HRM_HOME" "$HRM_USER"
fi

# Create installation directory
echo "Creating installation directory: $HRM_HOME"
sudo mkdir -p "$HRM_HOME"
sudo mkdir -p "$HRM_HOME/logs"
sudo mkdir -p "$HRM_HOME/backups"
sudo mkdir -p "$HRM_HOME/config"
sudo mkdir -p "/var/log/hrm"

# Copy systemd service file
echo "Installing systemd service..."
sudo cp hrm.service "$SERVICE_FILE"
sudo systemctl daemon-reload

# Copy binaries (assuming they're built)
echo "Installing HRM binaries..."
if [ -f "build/release/hrm_system" ]; then
    sudo cp build/release/hrm_system "$HRM_HOME/hrm_system"
    sudo cp build/release/hrm_vulkan "$HRM_HOME/hrm_vulkan"
else
    echo "Error: Built binaries not found. Please build first with:"
    echo "  mkdir build && cd build && cmake .. && make"
    exit 1
fi

# Set permissions
echo "Setting permissions..."
sudo chown -R "$HRM_USER:$HRM_USER" "$HRM_HOME"
sudo chown -R "$HRM_USER:$HRM_USER" "/var/log/hrm"
sudo chmod +x "$HRM_HOME/hrm_system"
sudo chmod +x "$HRM_HOME/hrm_vulkan"

# Create configuration
echo "Creating default configuration..."
sudo tee "$HRM_HOME/config/hrm_config.json" > /dev/null <<EOF
{
    "auto_boot": true,
    "idle_processing": true,
    "self_modification": true,
    "self_repair": true,
    "resource_monitoring": true,
    "log_level": "info",
    "backup_directory": "$HRM_HOME/backups",
    "max_rollback_points": 10,
    "protected_files": [
        "$HRM_HOME/hrm_system",
        "$HRM_HOME/config/hrm_config.json",
        "/etc/systemd/system/hrm.service"
    ]
}
EOF

sudo chown "$HRM_USER:$HRM_USER" "$HRM_HOME/config/hrm_config.json"

# Enable and start service
echo "Enabling HRM service..."
sudo systemctl enable "$HRM_SERVICE_NAME"
echo "Starting HRM service..."
sudo systemctl start "$HRM_SERVICE_NAME"

# Wait a moment for service to start
sleep 3

# Check service status
if sudo systemctl is-active --quiet "$HRM_SERVICE_NAME"; then
    echo "✅ HRM service started successfully!"
    echo "📊 Service status:"
    sudo systemctl status "$HRM_SERVICE_NAME" --no-pager -l
    echo ""
    echo "📋 Useful commands:"
    echo "  View logs:     sudo journalctl -u $HRM_SERVICE_NAME -f"
    echo "  Restart:        sudo systemctl restart $HRM_SERVICE_NAME"
    echo "  Stop:           sudo systemctl stop $HRM_SERVICE_NAME"
    echo "  Disable:         sudo systemctl disable $HRM_SERVICE_NAME"
    echo ""
    echo "🔧 Configuration file: $HRM_HOME/config/hrm_config.json"
    echo "📁 Log directory:   /var/log/hrm"
    echo "💾 Backup directory: $HRM_HOME/backups"
else
    echo "❌ Failed to start HRM service"
    echo "📋 Check logs with: sudo journalctl -u $HRM_SERVICE_NAME -f"
    exit 1
fi