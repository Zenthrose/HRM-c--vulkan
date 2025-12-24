#!/usr/bin/env bash

# HRM System End-to-End Test Script
# This script tests the complete HRM system functionality

# Parse arguments
VERBOSE=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --verbose|-v)
      VERBOSE=true
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --verbose, -v    Enable verbose output"
      echo "  --help, -h       Show this help"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

echo "🧪 HRM System End-to-End Testing"
echo "================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_RUN=0
TESTS_PASSED=0

run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_exit="$3"

    echo -n "Testing: $test_name... "
    TESTS_RUN=$((TESTS_RUN + 1))

    # Run the test
    if [ "$VERBOSE" = true ]; then
        eval "$test_command"
        local exit_code=$?
    else
        eval "$test_command" > /tmp/test_output.log 2>&1
        local exit_code=$?
    fi

    if [ "$expected_exit" = "success" ] && [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}PASSED${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    elif [ "$expected_exit" = "failure" ] && [ $exit_code -ne 0 ]; then
        echo -e "${GREEN}PASSED${NC} (expected failure)"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}FAILED${NC} (exit code: $exit_code)"
        if [ "$VERBOSE" = false ]; then
            echo "Output:"
            cat /tmp/test_output.log
        fi
    fi
}

# Check if we're in the build directory
if [ ! -f "./release/nyx_system" ]; then
    if [ -d "build" ]; then
        cd build
    fi
fi

# Check if executables exist
if [ ! -f "./release/nyx_system" ]; then
    echo -e "${RED}Error: nyx_system executable not found. Please build the project first.${NC}"
    exit 1
fi

if [ ! -f "./src/nyx_vulkan" ]; then
    echo -e "${YELLOW}Warning: nyx_vulkan executable not found. Some tests will be skipped.${NC}"
fi

echo "📋 Running System Tests..."
echo

# Test 1: Basic functionality test
run_test "Basic System Test" "./release/nyx_system --test" "success"

# Test 2: Configuration loading
run_test "Configuration Loading" "./release/nyx_system --test 2>/dev/null | grep -q 'Memory Compaction System initialized'" "success"

# Test 3: Help message
run_test "Help Message Display" "./release/nyx_system --help 2>/dev/null | grep -q 'HRM - Hierarchical Reasoning Module'" "success"

# Test 4: Invalid argument handling
run_test "Invalid Argument Handling" "./release/nyx_system --invalid 2>/dev/null" "failure"

# Test 5: Vulkan system still works
if [ -f "./release/nyx_vulkan" ]; then
    run_test "Vulkan System Compatibility" "timeout 10 ./release/nyx_vulkan 2>/dev/null | grep -q 'Vulkan instance created'" "success"
else
    echo -e "${YELLOW}Skipping Vulkan test - executable not found${NC}"
fi

# Test 6: Memory compaction functionality
run_test "Memory Compaction" "./release/nyx_system --test 2>/dev/null | grep -q 'Memory compaction successful'" "success"

# Test 7: Cloud storage functionality
run_test "Cloud Storage" "./release/nyx_system --test 2>/dev/null | grep -q 'Cloud storage initialized'" "success"

# Test 8: Configuration file exists
run_test "Configuration File" "[ -f '../config/hrm_config.txt' ]" "success"

echo
echo "📊 Test Results Summary:"
echo "========================"
echo "Tests Run: $TESTS_RUN"
echo "Tests Passed: $TESTS_PASSED"
echo "Success Rate: $((TESTS_PASSED * 100 / TESTS_RUN))%"

if [ $TESTS_PASSED -eq $TESTS_RUN ]; then
    echo -e "${GREEN}🎉 All tests passed! HRM system is ready for production use.${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Please check the output above for details.${NC}"
    exit 1
fi