# HRM Installation Guide

This guide covers installation and setup for the HRM (Hierarchical Reasoning Model) system on different platforms.

## Prerequisites

### All Platforms
- Vulkan 1.3+ compatible GPU
- CMake 3.16+
- C++17 compiler
- 4GB RAM minimum, 8GB+ recommended

### Linux
- GCC 9+ or Clang 10+
- Vulkan SDK or drivers

### macOS
- Xcode 12+
- Vulkan SDK (via MoltenVK)

### Windows
- MSYS2 with MinGW GCC 15.2.0
- Vulkan SDK

## Quick Install

### Linux/macOS
```bash
git clone https://github.com/Zenthrose/HRM-c--vulkan.git
cd HRM-c--vulkan
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Windows
```batch
git clone https://github.com/Zenthrose/HRM-c--vulkan.git
cd HRM-c--vulkan
set PATH=C:\msys64\mingw64\bin;%PATH%
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=C:\msys64\mingw64\bin\g++.exe -DCMAKE_C_COMPILER=C:\msys64\mingw64\bin\gcc.exe
cmake --build . --config Release -j 4
```

### Docker
```bash
docker build -t hrm .
docker run -it hrm
```

## Platform-Specific Setup

### Linux (Ubuntu/Debian)

1. Install Vulkan:
```bash
# AMD GPUs
sudo apt install mesa-vulkan-drivers

# NVIDIA GPUs
sudo apt install nvidia-driver-XXX vulkan-tools

# Intel GPUs
sudo apt install intel-media-va-driver-non-free
```

2. Install build tools:
```bash
sudo apt install build-essential cmake ninja-build
```

3. Install Vulkan SDK (optional, for development):
```bash
wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.3.250-jammy.list https://packages.lunarg.com/vulkan/1.3.250/lunarg-vulkan-1.3.250-jammy.list
sudo apt update && sudo apt install vulkan-sdk
```

### macOS

1. Install Xcode Command Line Tools:
```bash
xcode-select --install
```

2. Install Vulkan via Homebrew:
```bash
brew install molten-vk vulkan-headers
```

3. For Intel Macs, ensure Rosetta if needed.

### Windows

1. Install MSYS2 from https://www.msys2.org/

2. Update MSYS2:
```bash
pacman -Syu
```

3. Install MinGW GCC:
```bash
pacman -S mingw-w64-x86_64-gcc
```

4. Install Vulkan SDK from https://vulkan.lunarg.com/sdk/home

5. Add to PATH: `C:\msys64\mingw64\bin` and Vulkan SDK bin

## Environment Variables

Set these for custom paths:

```bash
export HRM_CONFIG_DIR=~/.hrm/config
export HRM_LOG_DIR=~/hrm/logs
export HRM_MODEL_DIR=~/hrm/models
export HRM_CLOUD_API_KEY=your_key
export HRM_LOG_LEVEL=info
```

## Troubleshooting

### Vulkan Issues

- Check Vulkan installation: `vulkaninfo`
- Ensure GPU drivers are up to date
- On Windows, verify Vulkan SDK is in PATH

### Build Issues

- Clear build cache: `rm -rf build/`
- Check CMake version: `cmake --version`
- Verify compiler: `g++ --version`

### Runtime Issues

- Check Vulkan compatibility: `./hrm_vulkan`
- Verify shader compilation
- Check log files in configured log directory

## Performance Tuning

- Use Release build for production
- Adjust batch sizes based on GPU memory
- Monitor with `nvidia-smi` or `radeontop`