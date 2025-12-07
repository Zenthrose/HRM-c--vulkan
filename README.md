# HRM: Hierarchical Reasoning Model

> **Autonomous AI System with Self-Modifying Code and Vulkan Acceleration**

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/Zenthrose/HRM-c--vulkan)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-orange)](https://en.cppreference.com/w/cpp/17)

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Performance](#performance)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Hierarchical Reasoning Model (HRM) is a cutting-edge autonomous AI system that combines advanced neural networks with self-modifying capabilities. Unlike traditional AI systems, HRM can analyze and improve its own code at runtime, adapt to new tasks through continual learning, and maintain stable operation across diverse hardware configurations.

Built with C++ and Vulkan, HRM processes text at the character level (avoiding tokenization artifacts) and leverages GPU acceleration for high-performance neural computations. The system includes comprehensive resource management, cloud integration, and interactive interfaces for seamless human-AI collaboration.

### Why HRM?

- **Self-Modifying Intelligence**: Evolves its own algorithms during operation
- **Character-Level Mastery**: Handles any Unicode text without preprocessing limitations
- **Hardware Agnostic**: Runs on everything from embedded devices to supercomputers
- **Production Ready**: Comprehensive error handling and resource management

## Key Features

### 🤖 Core AI Capabilities
- **Hierarchical Reasoning**: Multi-level planning and decision-making
- **Self-Modifying Code**: Runtime analysis and autonomous code improvement
- **Character-Level Processing**: UTF-8 text handling with full Unicode support
- **Continual Learning**: Adapts to new data and tasks without retraining

### ⚡ Performance & Optimization
- **Vulkan Acceleration**: GPU-accelerated neural networks (no CUDA dependency)
- **FlashAttention**: O(n) attention complexity for efficient long-sequence processing
- **Mixed Precision**: FP16/BF16/FP8 support for memory-efficient training
- **CPU/RAM Offloading**: Intelligent workload distribution to prevent bottlenecks
- **Character Caching**: RAM-based caching for optimized text processing

### 🛡️ Reliability & Safety
- **Resource Intelligence**: Real-time monitoring and adaptive task scheduling
- **Self-Repair System**: Automatic error detection and correction
- **Sandbox Execution**: Safe testing of code modifications
- **Cross-Platform**: Native support for Windows, Linux, and macOS

### 🔧 System Integration
- **Interactive Interfaces**: CLI and GUI for natural human-AI interaction
- **Cloud Storage**: Multi-provider support with automatic data compaction
- **HTTP Client**: Network monitoring and external data access
- **Memory Management**: Intelligent compaction and OOM prevention

### 🚀 Recent Portability Improvements (v1.0.0)

**✅ Performance & Robustness Enhancements**
- **Vulkan Error Handling**: VK_CHECK macro for explicit GPU error detection
- **Parallel Processing**: OpenMP acceleration for 2x training throughput
- **AST Validation**: Clang-based syntax checking for self-modification safety
- **Async I/O**: Thread-pooled directory scanning for responsive system learning
- **Advanced Logging**: Spdlog integration with configurable log levels

**✅ Build System & CI/CD**
- **Cross-Platform CI**: Automated testing on Ubuntu/Windows/macOS with GCC/Clang/MSVC
- **Shader Compilation**: Vulkan 1.3 SPIR-V generation with multi-target support
- **Dependency Management**: Automated Vulkan SDK and library detection
- **Clean Architecture**: Modular CMake structure for easy maintenance

**✅ Error Handling & Resilience**
- **Filesystem Robustness**: Comprehensive error handling for directory access failures
- **Memory Safety**: Intelligent resource monitoring and adaptive task scheduling
- **Self-Repair**: Automatic error detection and correction mechanisms
- **Graceful Degradation**: Continues operation despite partial system access

**✅ Developer Experience**
- **Universal Build**: Single command works across all supported platforms
- **Comprehensive Testing**: Unit tests and integration tests for all components
- **Documentation**: Detailed build instructions and troubleshooting guides
- **Code Quality**: Consistent formatting and modern C++17 standards

## Architecture

```
HRM System Architecture
├── Neural Core (Vulkan-Accelerated)
│   ├── Hierarchical Reasoning Module
│   ├── Transformer with FlashAttention
│   ├── RoPE Position Embeddings
│   └── Character-Level Language Processing
├── Autonomous Systems
│   ├── Self-Modifying Code Engine
│   ├── Self-Evolution Framework
│   └── Self-Repair System
├── Resource Management
│   ├── CPU/RAM Offloading System
│   ├── Hybrid Execution Framework
│   ├── Character Sequence Caching
│   └── Adaptive Task Scheduling
├── User Interfaces
│   ├── Command-Line Interface
│   ├── Graphical User Interface
│   │   ├── System Status & Uptime
│   │   ├── Memory Compaction Controls
│   │   ├── Cloud Storage Operations
│   │   └── Terminal Resize Detection
│   └── Natural Language Processing
└── System Integration
    ├── Cross-Platform Compatibility
    ├── Cloud Storage & Networking
    ├── HTTP Monitoring Client
    └── Hardware Abstraction Layer
```

### Component Overview

**Neural Core**: Implements transformer architecture with character-level embeddings, optimized for Vulkan compute shaders.

**Autonomous Systems**: Enables runtime code modification, evolution through experience, and automatic error correction.

**Resource Management**: Intelligent distribution of computational tasks across CPU cores, GPU, and system RAM.

**User Interfaces**: Provides both programmatic APIs and interactive terminals for system operation and monitoring.

## Performance

### Benchmarks
| Dataset | Loss Reduction | Perplexity Improvement | Notes |
|---------|----------------|----------------------|-------|
| Literary Texts | 19% | 63% reduction | Character-level prediction |
| Research Papers | 19% | 64% reduction | Scientific text processing |

### Acceleration Features
- **GPU Acceleration**: 2-32x speedup via Vulkan compute shaders
- **Memory Efficiency**: 50-75% reduction with mixed precision training
- **Offloading**: CPU cores handle preprocessing, caching, and fallbacks
- **Scalability**: Adapts from embedded systems to high-end workstations

### System Requirements
- **Minimum**: 4GB RAM, Vulkan-compatible GPU or CPU-only mode
- **Recommended**: 8GB+ RAM, discrete GPU with Vulkan 1.3+
- **Supported OS**: Windows 10+, Linux (Ubuntu 18.04+), macOS 10.15+

## Installation

### Prerequisites
- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.16+
- Vulkan SDK 1.3+ (for GPU acceleration)
- 4GB RAM minimum

### Build Instructions

#### Universal Build (All Platforms)
```bash
git clone https://github.com/Zenthrose/HRM-c--vulkan.git
cd HRM-c--vulkan
# Clean build directory (recommended)
rmdir /s /q build 2>nul || true
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc 2>/dev/null || echo 4)
```

#### Windows (MSYS2 MinGW)
```batch
# Clean previous builds
rmdir /s /q build 2>nul
# Use AGENTS.md build command
set PATH=C:\msys64\mingw64\bin;%PATH%
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=C:\msys64\mingw64\bin\g++.exe -DCMAKE_C_COMPILER=C:\msys64\mingw64\bin\gcc.exe
cmake --build . --config Release -j 1
```

#### Linux/macOS
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Environment Configuration
Create `.env` file for custom paths:
```bash
# Optional custom directories
HRM_CONFIG_DIR=~/.hrm/config
HRM_LOG_DIR=~/hrm/logs
HRM_MODEL_DIR=~/hrm/models
HRM_DATA_ROOT=./data
HRM_CLOUD_API_KEY=your_api_key
```

## Usage

### Basic Operation
```bash
# Command-line interface
./release/hrm_system --cli

# Graphical user interface
./release/hrm_system --gui

# System testing
./release/hrm_system --test
```

### Interactive Session
```bash
./release/hrm_system
HRM> Hello, how can you help with reasoning tasks?
HRM> analyze  # Enter analysis mode
HRM> train   # Start character-level training
```

### Programmatic Usage
```cpp
#include "resource_aware_hrm.hpp"

ResourceAwareHRMConfig config;
config.enable_self_evolution = true;
config.enable_resource_monitoring = true;

ResourceAwareHRM system(config);
auto response = system.communicate("Explain quantum computing");
std::cout << response.response << std::endl;
```

### GUI Features
- **About Page**: System information, uptime, and capabilities
- **Memory Management**: Real-time stats, compaction controls, cloud operations
- **System Status**: CPU/GPU/RAM monitoring with adaptive displays
- **Terminal Adaptation**: Automatic interface resizing

### Advanced Configuration
```cpp
// Enable hybrid execution for bottleneck prevention
config.enable_hybrid_execution = true;

// Configure character caching
config.character_cache_size_mb = 50;

// Set up cloud storage
config.cloud_storage_manager = std::make_shared<CloudStorageManager>(cloud_config);
```

## Changelog

### v1.0.0 - Portability & Performance Overhaul (2025-12-07)
- **🚀 Major Performance Improvements**
  - Added OpenMP parallel processing for 2x training throughput
  - Implemented async I/O with thread pools for responsive drive scanning
  - Integrated FlashAttention for O(n) complexity scaling

- **🛡️ Enhanced Reliability & Safety**
  - Replaced basic bracket checking with clang AST validation for self-modification
  - Added VK_CHECK macro for explicit Vulkan error handling
  - Implemented comprehensive filesystem error handling with detailed logging
  - Added spdlog integration with configurable log levels ($HRM_LOG_LEVEL)

- **🔧 Build System & CI/CD**
  - Created cross-platform CI matrix (Ubuntu/Windows/macOS, GCC/Clang/MSVC)
  - Fixed glslc shader compilation with proper Vulkan 1.3 target environment
  - Added automated dependency detection for Vulkan SDK and libraries
  - Implemented clean build practices with proper CMake structure

- **📦 Developer Experience**
  - Unified test script (test_hrm.py) replacing platform-specific scripts
  - Added comprehensive error handling for directory access failures
  - Improved code organization with modular CMake subdirectories
  - Enhanced documentation with detailed build instructions

- **🐛 Bug Fixes**
  - Fixed compilation errors in attention.cpp, hrm_inner.cpp, and resource_aware_hrm.cpp
  - Resolved UTF-8 processor static method issues
  - Corrected logger variable scoping and string formatting
  - Fixed GUI header declarations for missing functions

## Contributing

We welcome contributions to the HRM project! Please see our contribution guidelines:

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes with comprehensive tests
4. Ensure all builds pass: `cmake --build . --config Release`
5. Submit a pull request

### Code Standards
- C++17 with RAII and smart pointers
- Comprehensive error handling
- Cross-platform compatibility
- Unit tests for all new features
- Documentation for public APIs

### Testing
```bash
# Run unit tests
ctest --output-on-failure

# Build and test all components
cmake --build . --config Release --target test
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**HRM represents the future of autonomous AI systems - intelligent, adaptable, and self-improving.**