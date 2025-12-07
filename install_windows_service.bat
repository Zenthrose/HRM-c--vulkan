#!/bin/bash

# HRM Windows Service Installation Script
# Sets up HRM as a Windows service with auto-boot and idle processing

set -e

HRM_SERVICE_NAME="HRMSystem"
HRM_INSTALL_DIR="C:\\Program Files\\HRM"
HRM_EXECUTABLE="$HRM_INSTALL_DIR\\hrm_system.exe"
SERVICE_CONFIG="$HRM_INSTALL_DIR\\service_config.json"

echo "🚀 Installing HRM Self-Evolving AI System on Windows..."

# Check for administrator privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo "❌ Error: Administrator privileges required!"
    echo "Please run this script as Administrator."
    pause
    exit 1
)

# Create installation directory
echo "Creating installation directory: %HRM_INSTALL_DIR%"
if not exist "%HRM_INSTALL_DIR%" (
    mkdir "%HRM_INSTALL_DIR%"
)

# Copy binaries
echo "Installing HRM binaries..."
if exist "build\\release\\hrm_system.exe" (
    copy "build\\release\\hrm_system.exe" "%HRM_EXECUTABLE%"
    copy "build\\release\\hrm_vulkan.exe" "%HRM_INSTALL_DIR%\\hrm_vulkan.exe"
) else (
    echo "Error: Built binaries not found. Please build first."
    echo "Run: mkdir build && cd build && cmake .. && cmake --build . --config Release"
    pause
    exit 1
)

# Create configuration
echo "Creating service configuration..."
echo { > "%SERVICE_CONFIG%"
echo     "auto_boot": true, >> "%SERVICE_CONFIG%"
echo     "idle_processing": true, >> "%SERVICE_CONFIG%"
echo     "self_modification": true, >> "%SERVICE_CONFIG%"
echo     "self_repair": true, >> "%SERVICE_CONFIG%"
echo     "resource_monitoring": true, >> "%SERVICE_CONFIG%"
echo     "log_level": "info", >> "%SERVICE_CONFIG%"
echo     "backup_directory": "C:\\ProgramData\\HRM\\backups", >> "%SERVICE_CONFIG%"
echo     "max_rollback_points": 10, >> "%SERVICE_CONFIG%"
echo     "protected_files": [ >> "%SERVICE_CONFIG%"
echo         "%HRM_EXECUTABLE%", >> "%SERVICE_CONFIG%"
echo         "%SERVICE_CONFIG%", >> "%SERVICE_CONFIG%"
echo         "C:\\Windows\\System32\\config\\hrm.service" >> "%SERVICE_CONFIG%"
echo     ] >> "%SERVICE_CONFIG%"
echo } >> "%SERVICE_CONFIG%"

# Create backup directory
echo "Creating backup directory..."
if not exist "C:\\ProgramData\\HRM\\backups" (
    mkdir "C:\\ProgramData\\HRM"
    mkdir "C:\\ProgramData\\HRM\\backups"
)

# Install Windows service
echo "Installing Windows service..."
sc create "%HRM_SERVICE_NAME%" binPath= "%HRM_EXECUTABLE%" --service --daemon start= auto
sc description "%HRM_SERVICE_NAME%" "HRM Self-Evolving AI System with auto-boot and idle processing"
sc config "%HRM_SERVICE_NAME%" start= auto
sc config "%HRM_SERVICE_NAME%" type= own
sc config "%HRM_SERVICE_NAME%" error= normal
sc config "%HRM_SERVICE_NAME%" DisplayName= "HRM AI System"

# Set service dependencies
sc config "%HRM_SERVICE_NAME%" depend= Tcpip/Dnscache/LanmanServer/Netlogon

# Start the service
echo "Starting HRM service..."
sc start "%HRM_SERVICE_NAME%"

# Wait a moment for service to start
timeout /t 3 /nobreak >nul

# Check service status
sc query "%HRM_SERVICE_NAME%" | find "RUNNING" >nul
if %errorLevel% equ 0 (
    echo "✅ HRM service started successfully!"
    echo ""
    echo "📊 Service management commands:"
    echo "  View status:    sc query %HRM_SERVICE_NAME%"
    echo "  Restart:         sc restart %HRM_SERVICE_NAME%"
    echo "  Stop:            sc stop %HRM_SERVICE_NAME%"
    echo "  Uninstall:       sc delete %HRM_SERVICE_NAME%"
    echo ""
    echo "🔧 Configuration:   %SERVICE_CONFIG%"
    echo "📁 Install directory: %HRM_INSTALL_DIR%"
    echo "💾 Backup directory: C:\\ProgramData\\HRM\\backups"
    echo ""
    echo "📋 View logs: Check Windows Event Viewer under Applications and Services Logs"
) else (
    echo "❌ Failed to start HRM service"
    echo "📋 Check Windows Event Viewer for error details"
    echo "🔧 Manual start: sc start %HRM_SERVICE_NAME%"
    pause
    exit 1
)

echo ""
echo "🎉 HRM System installation complete!"
echo "The system will now automatically start on boot and process during idle times."