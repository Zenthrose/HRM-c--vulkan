# C++ Dependencies and Installation Guide
# This file lists all required programs and libraries to compile and run the HRM Vulkan system

## Cross-Platform Requirements
- CMake 3.16+ (https://cmake.org/download/)
- Vulkan SDK 1.3+ (https://vulkan.lunarg.com/sdk/home)
- Git (for cloning and submodules)

## Conda Environment Setup (Alternative)
For users preferring conda environments:
```bash
# Create HRM development environment
conda create -n hrm-dev python=3.10 -y
conda activate hrm-dev

# Install conda-forge packages
conda install -c conda-forge cmake ninja vulkan-headers vulkan-tools -y

# Note: Vulkan runtime libraries still need system installation
# On Linux: sudo apt install libvulkan1 mesa-vulkan-drivers
# On macOS: Vulkan SDK from LunarG
# On Windows: Vulkan SDK installer
```

## Linux (Ubuntu/Debian)
```bash
# Update package list
sudo apt update

# Install build tools
sudo apt install -y build-essential cmake ninja-build

# Install Vulkan development packages
sudo apt install -y vulkan-tools libvulkan-dev vulkan-validationlayers-dev

# Install additional development libraries
sudo apt install -y libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev

# Optional: Install clang for alternative compilation
sudo apt install -y clang
```

## Linux (Fedora/CentOS/RHEL)
```bash
# Update package list
sudo dnf update

# Install build tools
sudo dnf install -y gcc gcc-c++ cmake ninja-build

# Install Vulkan development packages
sudo dnf install -y vulkan-tools vulkan-loader-devel

# Install additional development libraries
sudo dnf install -y libX11-devel libXrandr-devel libXinerama-devel libXcursor-devel libXi-devel

# Optional: Install clang
sudo dnf install -y clang
```

## macOS
```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew (if not already installed)
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install build tools
brew install cmake ninja

# Install Vulkan SDK
# Download and install from: https://vulkan.lunarg.com/sdk/home
# Or use homebrew cask
brew install --cask vulkan-sdk

# Install additional libraries
brew install molten-vk
```

## Windows (MSYS2 MinGW)
```batch
# Download and install MSYS2 from: https://www.msys2.org/
# Run MSYS2 MSYS terminal and update
pacman -Syu

# Install MinGW-w64 toolchain
pacman -S mingw-w64-x86_64-toolchain

# Install build tools
pacman -S mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja

# Install Vulkan SDK
# Download from: https://vulkan.lunarg.com/sdk/home
# Extract to C:\VulkanSDK\1.3.x.x\

# Install additional libraries
pacman -S mingw-w64-x86_64-vulkan-devel

# Optional: Install LLVM/Clang
pacman -S mingw-w64-x86_64-clang
```

## Windows (Visual Studio - Alternative)
```batch
# Install Visual Studio 2022 Community with:
# - Desktop development with C++
# - Windows SDK
# - C++ CMake tools for Windows

# Install Vulkan SDK
# Download from: https://vulkan.lunarg.com/sdk/home

# Note: MSVC compilation may require additional code modifications
```

## Verification Commands

### Linux/macOS
```bash
# Check CMake
cmake --version

# Check Vulkan
vulkaninfo

# Check compiler
gcc --version
clang --version
```

### Windows (MSYS2)
```bash
# Check CMake
cmake --version

# Check Vulkan
vulkaninfo

# Check compiler
gcc --version
g++ --version
```

## Environment Setup

### Windows PATH Requirements
Ensure these are in your PATH:
- C:\msys64\mingw64\bin (MSYS2 MinGW)
- C:\VulkanSDK\1.3.x.x\Bin
- C:\Program Files\CMake\bin
- C:\ninja-win

### Linux/macOS PATH
Vulkan and CMake should be in PATH by default after installation.

## Build Verification
After installation, test the build:
```bash
git clone https://github.com/Zenthrose/HRM-c--vulkan.git
cd HRM-c--vulkan
mkdir build && cd build
cmake -G Ninja ..
cmake --build . --config Release
```

## Troubleshooting
- **Vulkan not found:** Ensure Vulkan SDK is properly installed and in PATH
- **CMake errors:** Update CMake to version 3.16+
- **Compiler issues:** Ensure C++17 support (GCC 7+, Clang 5+, MSVC 2017+)
- **Windows MSYS2:** Use mingw64 shell, not msys2 shell for compilation

## Optional Dependencies
- **Python 3.8+:** For running evaluation scripts and data processing
- **CUDA:** For GPU acceleration (if available)
- **NVIDIA drivers:** For Vulkan on NVIDIA GPUs</content>
<parameter name="filePath">cpp_requirements.txt