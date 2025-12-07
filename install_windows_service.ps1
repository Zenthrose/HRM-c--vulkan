# HRM Windows Service Installation Script (PowerShell)
# Sets up HRM as a Windows service with auto-boot and idle processing

param(
    [string]$InstallPath = "$env:ProgramFiles\HRM",
    [string]$ServiceName = "HRMSystem"
)

Write-Host "🚀 Installing HRM Self-Evolving AI System on Windows..." -ForegroundColor Cyan

# Check for administrator privileges
$currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "❌ Error: Administrator privileges required!" -ForegroundColor Red
    Write-Host "Please run this script as Administrator."
    Read-Host "Press Enter to exit"
    exit 1
}

$executable = Join-Path $InstallPath "hrm_system.exe"
$vulkanExecutable = Join-Path $InstallPath "hrm_vulkan.exe"
$serviceConfig = Join-Path $InstallPath "service_config.json"

# Create installation directory
Write-Host "Creating installation directory: $InstallPath"
if (-not (Test-Path $InstallPath)) {
    New-Item -ItemType Directory -Path $InstallPath -Force | Out-Null
}

# Copy binaries
Write-Host "Installing HRM binaries..."
$buildPath = Join-Path (Get-Location) "build\release"
if (Test-Path (Join-Path $buildPath "hrm_system.exe")) {
    Copy-Item (Join-Path $buildPath "hrm_system.exe") $executable
    Copy-Item (Join-Path $buildPath "hrm_vulkan.exe") $vulkanExecutable
} else {
    Write-Host "Error: Built binaries not found. Please build first." -ForegroundColor Red
    Write-Host "Run: mkdir build && cd build && cmake .. -DCMAKE_CXX_COMPILER=C:\msys64\mingw64\bin\g++.exe -DCMAKE_C_COMPILER=C:\msys64\mingw64\bin\gcc.exe && cmake --build . --config Release -j 1" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Create configuration
Write-Host "Creating service configuration..."
$configContent = @"
{
    "auto_boot": true,
    "idle_processing": true,
    "self_modification": true,
    "self_repair": true,
    "resource_monitoring": true,
    "log_level": "info",
    "backup_directory": "C:\\ProgramData\\HRM\\backups",
    "max_rollback_points": 10,
    "protected_files": [
        "$executable",
        "$serviceConfig",
        "C:\\Windows\\System32\\config\\hrm.service"
    ]
}
"@
$configContent | Out-File -FilePath $serviceConfig -Encoding UTF8

# Create backup directory
Write-Host "Creating backup directory..."
$backupDir = "C:\ProgramData\HRM"
if (-not (Test-Path $backupDir)) {
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
}
$backupPath = Join-Path $backupDir "backups"
if (-not (Test-Path $backupPath)) {
    New-Item -ItemType Directory -Path $backupPath -Force | Out-Null
}

# Install Windows service
Write-Host "Installing Windows service..."
try {
    $serviceParams = @{
        Name = $ServiceName
        BinaryPathName = "`"$executable`" --service --daemon"
        StartupType = "Automatic"
        DisplayName = "HRM AI System"
        Description = "HRM Self-Evolving AI System with auto-boot and idle processing"
    }

    New-Service @serviceParams

    # Set service dependencies
    sc.exe config $ServiceName depend= Tcpip/Dnscache/LanmanServer/Netlogon | Out-Null

    # Set error handling
    sc.exe config $ServiceName error= normal | Out-Null
    sc.exe config $ServiceName type= own | Out-Null

} catch {
    Write-Host "❌ Failed to create service: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Start the service
Write-Host "Starting HRM service..."
try {
    Start-Service -Name $ServiceName
    Start-Sleep -Seconds 3

    # Check service status
    $service = Get-Service -Name $ServiceName
    if ($service.Status -eq "Running") {
        Write-Host "✅ HRM service started successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "📊 Service management commands:" -ForegroundColor Cyan
        Write-Host "  View status:    Get-Service $ServiceName" -ForegroundColor White
        Write-Host "  Restart:        Restart-Service $ServiceName" -ForegroundColor White
        Write-Host "  Stop:           Stop-Service $ServiceName" -ForegroundColor White
        Write-Host "  Uninstall:      sc.exe delete $ServiceName" -ForegroundColor White
        Write-Host ""
        Write-Host "🔧 Configuration:   $serviceConfig" -ForegroundColor White
        Write-Host "📁 Install directory: $InstallPath" -ForegroundColor White
        Write-Host "💾 Backup directory: $backupPath" -ForegroundColor White
        Write-Host ""
        Write-Host "📋 View logs: Check Windows Event Viewer under Applications and Services Logs" -ForegroundColor White
    } else {
        Write-Host "❌ Failed to start HRM service" -ForegroundColor Red
        Write-Host "📋 Check Windows Event Viewer for error details" -ForegroundColor Yellow
        Write-Host "🔧 Manual start: Start-Service $ServiceName" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "❌ Failed to start service: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🎉 HRM System installation complete!" -ForegroundColor Green
Write-Host "The system will now automatically start on boot and process during idle times."