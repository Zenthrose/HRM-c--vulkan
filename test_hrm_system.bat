@echo off
REM HRM System End-to-End Test Script for Windows
REM This script tests the complete HRM system functionality

echo 🧪 HRM System End-to-End Testing
echo =================================

REM Colors for output (Windows CMD)
REM Note: Windows CMD has limited color support, using simple text

REM Test counter
set TESTS_RUN=0
set TESTS_PASSED=0

REM Check if we're in the build directory
if not exist "src\hrm_system.exe" (
    if exist "hrm_system.exe" (
        REM We're in src directory
        cd ..
    ) else if exist "build" (
        cd build
    )
)

REM Check if executables exist
if not exist "src\hrm_system.exe" (
    echo Error: hrm_system.exe executable not found. Please build the project first.
    exit /b 1
)

if not exist "src\hrm_vulkan.exe" (
    echo Warning: hrm_vulkan.exe executable not found. Some tests will be skipped.
)

echo 📋 Running System Tests...
echo.

REM Test 1: Basic functionality test
set /a TESTS_RUN+=1
echo Testing: Basic System Test...
src\hrm_system.exe --test > temp_output.log 2>&1
if %errorlevel% equ 0 (
    echo PASSED
    set /a TESTS_PASSED+=1
) else (
    echo FAILED (exit code: %errorlevel%)
    echo Output:
    type temp_output.log
)

REM Test 2: Configuration loading
set /a TESTS_RUN+=1
echo Testing: Configuration Loading...
src\hrm_system.exe --test 2>nul | findstr /C:"Memory Compaction System initialized" >nul
if %errorlevel% equ 0 (
    echo PASSED
    set /a TESTS_PASSED+=1
) else (
    echo FAILED
)

REM Test 3: Help message
set /a TESTS_RUN+=1
echo Testing: Help Message Display...
src\hrm_system.exe --help 2>nul | findstr /C:"HRM - Hierarchical Reasoning Module" >nul
if %errorlevel% equ 0 (
    echo PASSED
    set /a TESTS_PASSED+=1
) else (
    echo FAILED
)

REM Test 4: Invalid argument handling
set /a TESTS_RUN+=1
echo Testing: Invalid Argument Handling...
src\hrm_system.exe --invalid 2>nul >nul
if %errorlevel% neq 0 (
    echo PASSED (expected failure)
    set /a TESTS_PASSED+=1
) else (
    echo FAILED (should have failed)
)

REM Test 5: Vulkan system compatibility
if exist "src\hrm_vulkan.exe" (
    set /a TESTS_RUN+=1
    echo Testing: Vulkan System Compatibility...
    timeout /t 10 /nobreak >nul 2>&1
    start /b src\hrm_vulkan.exe 2>nul | findstr /C:"Vulkan instance created" >nul
    if %errorlevel% equ 0 (
        echo PASSED
        set /a TESTS_PASSED+=1
    ) else (
        echo FAILED
    )
) else (
    echo Skipping Vulkan test - executable not found
)

REM Test 6: Memory compaction functionality
set /a TESTS_RUN+=1
echo Testing: Memory Compaction...
src\hrm_system.exe --test 2>nul | findstr /C:"Memory compaction successful" >nul
if %errorlevel% equ 0 (
    echo PASSED
    set /a TESTS_PASSED+=1
) else (
    echo FAILED
)

REM Test 7: Cloud storage functionality
set /a TESTS_RUN+=1
echo Testing: Cloud Storage...
src\hrm_system.exe --test 2>nul | findstr /C:"Cloud storage initialized" >nul
if %errorlevel% equ 0 (
    echo PASSED
    set /a TESTS_PASSED+=1
) else (
    echo FAILED
)

REM Test 8: Configuration file exists
set /a TESTS_RUN+=1
echo Testing: Configuration File...
if exist "..\config\hrm_config.txt" (
    echo PASSED
    set /a TESTS_PASSED+=1
) else (
    echo FAILED
)

echo.
echo 📊 Test Results Summary:
echo ========================
echo Tests Run: %TESTS_RUN%
echo Tests Passed: %TESTS_PASSED%
set /a SUCCESS_RATE=TESTS_PASSED*100/TESTS_RUN
echo Success Rate: %SUCCESS_RATE%%%

if %TESTS_PASSED% equ %TESTS_RUN% (
    echo 🎉 All tests passed! HRM system is ready for production use.
    exit /b 0
) else (
    echo ❌ Some tests failed. Please check the output above for details.
    exit /b 1
)

REM Clean up
if exist temp_output.log del temp_output.log