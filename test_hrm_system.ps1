# HRM System End-to-End Test Script for Windows (PowerShell)
# This script tests the complete HRM system functionality

param(
    [switch]$Verbose
)

Write-Host "HRM System End-to-End Testing" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Test counter
$testsRun = 0
$testsPassed = 0

# Function to run test with verbose output
function Run-Test {
    param(
        [string]$testName,
        [scriptblock]$testScript,
        [bool]$expectFailure = $false
    )

    $script:testsRun++
    Write-Host "Testing: $testName..." -NoNewline

    try {
        $result = & $testScript
        $exitCode = $LASTEXITCODE

        if ($Verbose) {
            Write-Host ""
            Write-Host "Exit code: $exitCode" -ForegroundColor Gray
            if ($result) {
                Write-Host "Output: $result" -ForegroundColor Gray
            }
        }

        if (($expectFailure -and $exitCode -ne 0) -or (-not $expectFailure -and $exitCode -eq 0)) {
            Write-Host " PASSED" -ForegroundColor Green
            $script:testsPassed++
        } else {
            Write-Host " FAILED (exit code: $exitCode)" -ForegroundColor Red
            if ($Verbose -and $result) {
                Write-Host "Output:" -ForegroundColor Yellow
                Write-Host $result -ForegroundColor Yellow
            }
        }
    } catch {
        Write-Host " FAILED (exception: $($_.Exception.Message))" -ForegroundColor Red
        if ($Verbose) {
            Write-Host "Exception details: $($_.Exception)" -ForegroundColor Yellow
        }
    }
}

# Check if we're in the build directory
$currentPath = Get-Location
if (-not (Test-Path "src\hrm_system.exe")) {
    if (Test-Path "hrm_system.exe") {
        # We're in src directory
        Set-Location ..
    } elseif (Test-Path "build") {
        Set-Location build
    }
}

# Check if executables exist
if (-not (Test-Path "src\hrm_system.exe")) {
    Write-Host "Error: hrm_system.exe executable not found. Please build the project first." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "src\hrm_vulkan.exe")) {
    Write-Host "Warning: hrm_vulkan.exe executable not found. Some tests will be skipped." -ForegroundColor Yellow
}

Write-Host "Running System Tests..."
Write-Host ""

# Test 1: Basic functionality test
Run-Test "Basic System Test" {
    $output = & "src\hrm_system.exe" --test 2>&1
    if ($Verbose) { $output }
    $output
}

# Test 2: Configuration loading
Run-Test "Configuration Loading" {
    $output = & "src\hrm_system.exe" --test 2>$null
    if ($output -match "Memory Compaction System initialized") { $true } else { $false }
}

# Test 3: Help message
Run-Test "Help Message Display" {
    $output = & "src\hrm_system.exe" --help 2>$null
    if ($output -match "HRM - Hierarchical Reasoning Module") { $true } else { $false }
}

# Test 4: Invalid argument handling
Run-Test "Invalid Argument Handling" {
    & "src\hrm_system.exe" --invalid 2>$null >$null
} $true

# Test 5: Vulkan system compatibility
if (Test-Path "src\hrm_vulkan.exe") {
    Run-Test "Vulkan System Compatibility" {
        $job = Start-Job -ScriptBlock {
            & "src\hrm_vulkan.exe" 2>$null
        }
        Start-Sleep -Seconds 5
        $output = Receive-Job $job 2>$null
        Stop-Job $job
        Remove-Job $job
        if ($output -match "Vulkan instance created") { $true } else { $false }
    }
} else {
    Write-Host "Skipping Vulkan test - executable not found" -ForegroundColor Yellow
}

# Test 6: Memory compaction functionality
Run-Test "Memory Compaction" {
    $output = & "src\hrm_system.exe" --test 2>$null
    if ($output -match "Memory compaction successful") { $true } else { $false }
}

# Test 7: Cloud storage functionality
Run-Test "Cloud Storage" {
    $output = & "src\hrm_system.exe" --test 2>$null
    if ($output -match "Cloud storage initialized") { $true } else { $false }
}

# Test 8: Configuration file exists
Run-Test "Configuration File" {
    Test-Path "..\config\hrm_config.txt"
}

Write-Host ""
Write-Host "Test Results Summary:" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan
Write-Host "Tests Run: $testsRun"
Write-Host "Tests Passed: $testsPassed"
$successRate = if ($testsRun -gt 0) { [math]::Round(($testsPassed / $testsRun) * 100, 2) } else { 0 }
Write-Host "Success Rate: $successRate%"

if ($testsPassed -eq $testsRun) {
    Write-Host "All tests passed! HRM system is ready for production use." -ForegroundColor Green
    exit 0
} else {
    Write-Host "Some tests failed. Please check the output above for details." -ForegroundColor Red
    exit 1
}

# Clean up
if (Test-Path "temp_output.log") {
    Remove-Item "temp_output.log"
}