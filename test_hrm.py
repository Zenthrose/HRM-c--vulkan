#!/usr/bin/env python3
"""
HRM System Test Runner
Cross-platform test execution script that replaces separate .bat/.sh/.ps1 files
"""

import subprocess
import sys
import os
import argparse
import platform

def run_command(cmd, cwd=None, check=True):
    """Run a command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, check=check)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def find_executable(name):
    """Find executable in common locations."""
    # Check current directory build
    build_paths = [
        "build/release",
        "build/Release",
        "build",
        "."
    ]

    for path in build_paths:
        exe_path = os.path.join(path, name)
        if platform.system() == "Windows":
            exe_path += ".exe"
        if os.path.exists(exe_path):
            return exe_path

    return name  # Assume it's in PATH

def run_tests(verbose=False):
    """Run the HRM system tests."""
    print("🚀 Running HRM System Tests...")

    # Find test executable
    test_exe = find_executable("hrm_system_test")
    if not os.path.exists(test_exe):
        print(f"❌ Test executable not found: {test_exe}")
        print("Please build the project first: mkdir build && cd build && cmake .. && cmake --build . --config Release")
        return False

    # Run tests
    cmd = f'"{test_exe}"'
    if verbose:
        cmd += " --verbose"

    print(f"📋 Executing: {cmd}")
    success, stdout, stderr = run_command(cmd)

    if success:
        print("✅ Tests passed!")
        if verbose and stdout:
            print("📄 Test Output:")
            print(stdout)
    else:
        print("❌ Tests failed!")
        if stderr:
            print("📄 Error Output:")
            print(stderr)
        if stdout:
            print("📄 Standard Output:")
            print(stdout)

    return success

def run_system_test(verbose=False):
    """Run basic system functionality test."""
    print("🔧 Running HRM System Functionality Test...")

    # Find main executable
    exe = find_executable("hrm_system")
    if not os.path.exists(exe):
        print(f"❌ HRM executable not found: {exe}")
        print("Please build the project first")
        return False

    # Run basic test
    cmd = f'"{exe}" --test'
    if verbose:
        print(f"📋 Executing: {cmd}")

    success, stdout, stderr = run_command(cmd)

    if success:
        print("✅ System test passed!")
        if verbose and stdout:
            print("📄 Output:")
            print(stdout)
    else:
        print("❌ System test failed!")
        if stderr:
            print("📄 Error Output:")
            print(stderr)

    return success

def main():
    parser = argparse.ArgumentParser(description="HRM System Test Runner")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--unit-tests", action="store_true", help="Run unit tests only")
    parser.add_argument("--system-test", action="store_true", help="Run system functionality test only")

    args = parser.parse_args()

    print(f"🖥️  HRM Test Runner - {platform.system()} {platform.machine()}")
    print(f"🐍 Python {sys.version.split()[0]}")
    print()

    success = True

    if args.system_test:
        success &= run_system_test(args.verbose)
    elif args.unit_tests:
        success &= run_tests(args.verbose)
    else:
        # Run both by default
        success &= run_system_test(args.verbose)
        success &= run_tests(args.verbose)

    print()
    if success:
        print("🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()