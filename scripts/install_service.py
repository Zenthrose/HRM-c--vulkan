#!/usr/bin/env python3
"""
HRM Service Installer
Cross-platform service installation for HRM system
Supports Windows (NSSM), Linux (systemd), and macOS (launchd)
"""

import subprocess
import sys
import os
import platform
import argparse
from pathlib import Path

def run_command(cmd, check=True, shell=True):
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, shell=shell, check=check, capture_output=True, text=True)
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def detect_os():
    """Detect the operating system."""
    system = platform.system().lower()
    if system == "windows":
        return "windows"
    elif system == "linux":
        return "linux"
    elif system == "darwin":
        return "macos"
    else:
        return "unknown"

def check_admin():
    """Check if running with administrator privileges."""
    try:
        if detect_os() == "windows":
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin()
        else:
            return os.geteuid() == 0
    except AttributeError:
        return False

def install_windows_service(install_dir, service_name):
    """Install HRM as Windows service using NSSM or native commands."""
    print("🔧 Installing HRM as Windows service...")

    # Check for NSSM first
    nssm_paths = [
        "C:\\nssm\\nssm.exe",
        "C:\\Program Files\\nssm\\nssm.exe",
        "C:\\Program Files (x86)\\nssm\\nssm.exe"
    ]

    nssm_exe = None
    for path in nssm_paths:
        if os.path.exists(path):
            nssm_exe = path
            break

    exe_path = os.path.join(install_dir, "hrm_system.exe")

    if nssm_exe:
        print(f"📦 Using NSSM: {nssm_exe}")
        # Install with NSSM
        cmds = [
            f'"{nssm_exe}" install {service_name} "{exe_path}" --daemon',
            f'"{nssm_exe}" set {service_name} DisplayName "HRM AI System"',
            f'"{nssm_exe}" set {service_name} Description "Hierarchical Reasoning Model AI System"',
            f'"{nssm_exe}" set {service_name} Start SERVICE_AUTO_START',
            f'"{nssm_exe}" start {service_name}'
        ]
    else:
        print("⚠️  NSSM not found, using native sc.exe (limited functionality)")
        # Fallback to native Windows service (requires pre-compiled service)
        cmds = [
            f'sc.exe create {service_name} binPath= "{exe_path} --daemon" start= auto',
            f'sc.exe description {service_name} "Hierarchical Reasoning Model AI System"',
            f'sc.exe start {service_name}'
        ]

    for cmd in cmds:
        success, stdout, stderr = run_command(cmd)
        if not success:
            print(f"❌ Command failed: {cmd}")
            if stderr:
                print(f"Error: {stderr}")
            return False

    print(f"✅ HRM service '{service_name}' installed and started on Windows")
    return True

def install_linux_service(install_dir, service_name):
    """Install HRM as systemd service on Linux."""
    print("🔧 Installing HRM as systemd service...")

    service_content = f"""[Unit]
Description=HRM AI System
After=network.target

[Service]
Type=simple
User={os.getenv('USER', 'root')}
ExecStart={install_dir}/hrm_system --daemon
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""

    service_path = f"/etc/systemd/system/{service_name}.service"

    # Write service file
    try:
        with open(service_path, 'w') as f:
            f.write(service_content)
    except PermissionError:
        print("❌ Permission denied. Run with sudo.")
        return False

    # Reload systemd and enable/start service
    cmds = [
        "systemctl daemon-reload",
        f"systemctl enable {service_name}",
        f"systemctl start {service_name}"
    ]

    for cmd in cmds:
        success, stdout, stderr = run_command(cmd)
        if not success:
            print(f"❌ Command failed: {cmd}")
            if stderr:
                print(f"Error: {stderr}")
            return False

    print(f"✅ HRM service '{service_name}' installed and started on Linux")
    return True

def install_macos_service(install_dir, service_name):
    """Install HRM as launchd service on macOS."""
    print("🔧 Installing HRM as launchd service...")

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{service_name}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{install_dir}/hrm_system</string>
        <string>--daemon</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/var/log/hrm.log</string>
    <key>StandardErrorPath</key>
    <string>/var/log/hrm.error.log</string>
</dict>
</plist>
"""

    plist_path = f"/Library/LaunchDaemons/{service_name}.plist"

    # Write plist file
    try:
        with open(plist_path, 'w') as f:
            f.write(plist_content)
    except PermissionError:
        print("❌ Permission denied. Run with sudo.")
        return False

    # Load and start service
    cmds = [
        f"launchctl load {plist_path}",
        f"launchctl start {service_name}"
    ]

    for cmd in cmds:
        success, stdout, stderr = run_command(cmd)
        if not success:
            print(f"❌ Command failed: {cmd}")
            if stderr:
                print(f"Error: {stderr}")
            return False

    print(f"✅ HRM service '{service_name}' installed and started on macOS")
    return True

def main():
    parser = argparse.ArgumentParser(description="HRM Service Installer")
    parser.add_argument("--install-dir", default="C:\\Program Files\\HRM" if detect_os() == "windows" else "/usr/local/hrm",
                       help="Installation directory")
    parser.add_argument("--service-name", default="HRMSystem", help="Service name")
    parser.add_argument("--uninstall", action="store_true", help="Uninstall the service instead")

    args = parser.parse_args()

    os_type = detect_os()
    print(f"🖥️  HRM Service Installer - {os_type}")
    print(f"📁 Install Directory: {args.install_dir}")
    print(f"🏷️  Service Name: {args.service_name}")
    print()

    if args.uninstall:
        print("🗑️  Uninstall functionality not yet implemented")
        return

    # Check privileges
    if not check_admin():
        print("❌ Administrator/root privileges required!")
        if os_type == "windows":
            print("Right-click and 'Run as administrator'")
        else:
            print("Run with sudo")
        sys.exit(1)

    # Check if HRM is built
    exe_name = "hrm_system.exe" if os_type == "windows" else "hrm_system"
    exe_path = os.path.join(args.install_dir, exe_name)
    if not os.path.exists(exe_path):
        print(f"❌ HRM executable not found: {exe_path}")
        print("Please build HRM first and ensure it's in the install directory")
        sys.exit(1)

    # Install based on OS
    success = False
    if os_type == "windows":
        success = install_windows_service(args.install_dir, args.service_name)
    elif os_type == "linux":
        success = install_linux_service(args.install_dir, args.service_name)
    elif os_type == "macos":
        success = install_macos_service(args.install_dir, args.service_name)
    else:
        print(f"❌ Unsupported OS: {os_type}")
        sys.exit(1)

    if success:
        print()
        print("🎉 HRM service installation complete!")
        print("The system will automatically start on boot.")
    else:
        print()
        print("💥 Service installation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()