# Hierarchical Reasoning Model (HRM)

![HRM Logo](./assets/hrm.png)

> **Ultimate Autonomous AI System**: Self-modifying, self-repairing, resource-aware AI with advanced character-level language processing and Vulkan-accelerated neural networks.

## Table of Contents
- [🚀 Overview](#-overview)
- [🎯 Key Features](#-key-features)
- [📊 Implementation Status](#-implementation-status)
- [⚡ Quick Start](#-quick-start)
- [🛠️ Installation](#️-installation)
- [🎮 Usage](#-usage)
- [🏗️ Architecture](#️-architecture)
- [📈 Performance](#-performance)
- [🔬 Research & Citation](#-research--citation)
- [🐍 Original Python Implementation](#-original-python-implementation)

## 🚀 Overview

The Hierarchical Reasoning Model (HRM) is a cutting-edge AI system that combines:

- **Self-Modifying Intelligence**: Runtime code analysis and autonomous improvement
- **Vulkan-Accelerated Training**: Pure GPU neural network training without CUDA dependencies
- **Character-Level Language Processing**: UTF-8 text handling for multilingual capabilities
- **Resource-Aware Operation**: Adaptive task management and OOM prevention
- **Cross-Platform Compatibility**: Runs on Windows, Linux, macOS, and embedded systems

Built for the future of AI, HRM represents a complete autonomous learning system capable of modifying its own code, evolving through experience, and maintaining production stability across all major platforms.

## 🎯 Key Features

### Core AI Capabilities
- 🧠 **Self-Modifying Code**: Runtime analysis and rewriting of C++ source code
- 🔄 **Self-Evolving Intelligence**: Continual adaptation and meta-learning
- 🛡️ **Self-Repair System**: Automatic error detection and correction
- 💬 **Conversational AI**: Natural dialogue with context awareness
- 🎓 **Character-Level Language**: UTF-8 processing without tokenization overhead

### Advanced Training & Optimization
- ⚡ **FlashAttention**: O(n) complexity attention, 2-32x faster than standard
- 🎯 **Mixed Precision**: FP16/BF16/FP8 support with 50-75% memory reduction
- 🧮 **Advanced Optimizers**: Lion, Adafactor beyond AdamW
- 📊 **Gradient Checkpointing**: Memory-efficient training for large models
- 🔄 **Linear Attention**: Alternative O(n) attention mechanism

### System Integration
- 🔋 **Resource Intelligence**: Real-time monitoring and adaptive scheduling
- ☁️ **Cloud Storage**: Memory compaction with multi-provider support
- 🎮 **Interactive Interfaces**: CLI and GUI for human-AI communication
- 🌐 **Network Monitoring**: HTTP client for traffic analysis
- 🔧 **Cross-Platform**: Universal hardware support from embedded to supercomputers
- 🧠 **CPU/RAM Offloading**: Intelligent workload distribution to prevent bottlenecking
- 💾 **Character Sequence Caching**: RAM-based caching for efficient processing
- ⚖️ **Hybrid Execution**: Dynamic CPU/GPU task balancing with resource monitoring

## 📊 Implementation Status

| Component | Status | Description |
|-----------|--------|-------------|
| **Core Neural Networks** | ✅ Complete | HRM architecture with Vulkan acceleration |
| **Self-Modifying System** | ✅ Complete | Runtime code modification and hot-swapping |
| **Self-Evolution Framework** | ✅ Complete | Meta-learning and continual adaptation |
| **Self-Repair Mechanisms** | ✅ Complete | Error detection and automatic correction |
| **Character-Level Processing** | ✅ Complete | UTF-8 encoding/decoding with Unicode support |
| **Advanced Training Optimizations** | ✅ Complete | FlashAttention, mixed precision, optimizers |
| **Resource Management** | ✅ Complete | Real-time monitoring and OOM prevention |
| **Cross-Platform Compatibility** | ✅ Complete | Windows/Linux/macOS with Vulkan |
| **Network & Cloud Integration** | ✅ Complete | HTTP client and multi-provider storage |
| **Interactive Interfaces** | ✅ Complete | CLI/GUI with natural language processing |
| **Successful Compilation** | ✅ Complete | Builds without errors using standard commands |
| **GUI About Page & Uptime** | ✅ Complete | System information display with real-time uptime tracking |
| **Memory Compaction Controls** | ✅ Complete | GUI interface for memory management and compaction |
| **Cloud Storage Interface** | ✅ Complete | Full cloud operations with background processing |
| **Terminal Resize Detection** | ✅ Complete | Automatic GUI redraw on terminal size changes |
| **CPU/RAM Offloading System** | ✅ Complete | Intelligent workload distribution to prevent bottlenecking |
| **Character Sequence Caching** | ✅ Complete | RAM-based caching for efficient character processing |
| **Hybrid Execution Framework** | ✅ Complete | Dynamic CPU/GPU task balancing with resource monitoring |

### Platform Compatibility
- **Windows**: Full MSYS2 MinGW GCC support with Windows API integration
- **Linux/macOS**: GCC/Clang with system API integration
- **Hardware**: Any Vulkan-compatible GPU or CPU fallback
- **Build System**: CMake + Ninja with portable configuration

## ⚡ Quick Start

### Prerequisites
- Vulkan SDK 1.3+
- CMake 3.16+
- C++17 compiler (GCC/Clang/MSVC)
- 4GB RAM minimum, 8GB+ recommended

### Build & Run (Universal)
```bash
git clone https://github.com/Zenthrose/HRM-c--vulkan.git
cd HRM-c--vulkan
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)  # or cmake --build . --config Release -j 4 on Windows

# Run interfaces
./src/hrm_system --cli    # Command-line interface
./src/hrm_system --gui    # Graphical interface
./src/hrm_system --test   # System tests
```

### Conversational AI Demo
```bash
./src/hrm_system
HRM> Hello, how can you help with brainstorming?
HRM> train  # Start character-level training
```

### Enhanced GUI Features
The HRM GUI now provides comprehensive system management:
```bash
./src/hrm_system --gui
```
- **About Page**: System information, real-time uptime, capabilities overview
- **Memory Management**: Statistics, compaction controls, cloud storage operations
- **System Status**: CPU/RAM/GPU monitoring with uptime tracking
- **Terminal Resize**: Automatic interface adaptation to terminal size changes
- **Background Operations**: Async memory compaction and cloud transfers

## 🛠️ Installation

### Environment Setup
```bash
# Optional: Configure custom paths
export HRM_CONFIG_DIR=~/.hrm/config
export HRM_LOG_DIR=~/hrm/logs
export HRM_MODEL_DIR=~/hrm/models
```

### Windows (MSYS2 MinGW)
```batch
set PATH=C:\msys64\mingw64\bin;%PATH%
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=g++.exe -DCMAKE_C_COMPILER=gcc.exe
mingw32-make -j4
```

### Linux/macOS
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Docker
```bash
docker build -t hrm .
docker run -it hrm ./hrm_system --cli
```

## 🎮 Usage

### Basic Interaction
```cpp
#include "resource_aware_hrm.hpp"

ResourceAwareHRMConfig config;
config.enable_self_evolution = true;
config.enable_resource_monitoring = true;

ResourceAwareHRM system(config);
auto response = system.communicate("Explain quantum computing");
std::cout << response.response << std::endl;
```

### Advanced Training
```cpp
#include "character_language_trainer.hpp"

CharacterLanguageTrainer trainer(model, config);
trainer.train_character_language_model("training_corpus.txt");
```

### Self-Modification
```cpp
SelfModifyingHRM system(config);
// System automatically analyzes and improves its code
auto result = system.analyze_and_modify_self();
```

### Resource Management
```cpp
auto usage = system.get_current_resource_usage();
auto alerts = system.get_resource_alerts();
// Automatic adaptation to system constraints
```

## 🏗️ Architecture

```
HRM System Architecture
├── Core Neural Networks (Vulkan-Accelerated)
│   ├── Hierarchical Reasoning Module (H/L levels)
│   ├── Adaptive Computation Time (ACT)
│   ├── Transformer with FlashAttention
│   ├── Character-Level Language Processing
│   └── RoPE Position Embeddings
├── Autonomous Systems
│   ├── Self-Modifying Code Engine
│   ├── Self-Evolution Framework
│   └── Self-Repair System
├── Resource Intelligence
│   ├── Real-Time Monitoring
│   ├── Adaptive Task Scheduling
│   ├── OOM Prevention
│   ├── CPU/RAM Offloading
│   └── Hybrid Execution Framework
├── User Interfaces
│   ├── Command-Line Interface
│   ├── Graphical User Interface (Enhanced)
│   │   ├── About Page & System Info
│   │   ├── Memory Compaction Controls
│   │   ├── Cloud Storage Operations
│   │   └── Terminal Resize Detection
│   └── Natural Language Processing
├── Caching & Optimization
│   ├── Character Sequence Cache (RAM)
│   ├── Memory Compaction System
│   └── Intelligent Workload Balancing
└── System Integration
    ├── Cross-Platform Compatibility
    ├── Cloud Storage & Networking
    ├── HTTP Client for Monitoring
    └── Hardware Abstraction Layer
```

### Key Components

#### Neural Network Core
- **Hierarchical Reasoning**: Multi-level planning and execution
- **Vulkan Acceleration**: GPU compute shaders for training/inference
- **Character Processing**: UTF-8 handling with Unicode support
- **Optimization Stack**: FlashAttention, mixed precision, advanced optimizers

#### Autonomous Systems
- **Self-Modification**: Runtime code analysis and patching
- **Self-Evolution**: Learning from interactions and data
- **Self-Repair**: Error detection and automatic correction
- **Sandbox Testing**: Safe validation of modifications

#### System Integration
- **Resource Monitoring**: CPU, memory, GPU, network tracking
- **Cloud Storage**: Multi-provider with memory compaction
- **Network Client**: HTTP-based monitoring and communication
- **Hardware Abstraction**: Universal device compatibility

## 📈 Performance

### Training Benchmarks
| Dataset | Loss Reduction | Perplexity | Accuracy |
|---------|----------------|------------|----------|
| Pride & Prejudice | 19% (5.39→4.38) | 63% (219→80) | Character-level |
| ArXiv Research Papers | 19% (5.22→4.20) | 64% (184→67) | 2.4% prediction |

### Acceleration Features
- **FlashAttention**: 2-32x speedup, O(n) vs O(n²) complexity
- **Mixed Precision**: 50-75% memory savings with FP16/BF16/FP8
- **Vulkan Optimization**: Zero CUDA dependency, universal GPU support
- **Character-Level Efficiency**: No tokenization overhead
- **CPU/RAM Offloading**: Intelligent workload distribution prevents bottlenecking
- **Hybrid Execution**: Dynamic CPU/GPU balancing maximizes hardware utilization
- **Sequence Caching**: RAM-based caching reduces redundant character processing

### System Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 1 core | 8+ cores |
| RAM | 256MB | 8GB+ |
| GPU | None (CPU mode) | Vulkan-compatible |
| Storage | 500MB | 10GB+ |
| OS | Any | Linux/macOS/Windows |

## 🔬 Research & Citation

### Key Innovations
- **Hierarchical Reasoning**: Multi-timescale planning for complex tasks
- **Self-Modifying AI**: Runtime code improvement and evolution
- **Vulkan Neural Networks**: GPU acceleration without CUDA
- **Character-Level Mastery**: UTF-8 processing for universal language support
- **Resource-Aware Autonomy**: Adaptive operation across hardware constraints

### Citation
```bibtex
@misc{hrm_2024,
  title={Hierarchical Reasoning Model with Self-Modifying Code},
  author={Zenthrose},
  year={2024},
  url={https://github.com/Zenthrose/HRM-c--vulkan}
}
```

## 🐍 Original Python Implementation

The HRM was originally implemented in Python/PyTorch. See the [Python README section](#python-implementation-details) for the original research implementation, training scripts, and evaluation tools.

### Python Quick Start
```bash
pip install -r requirements.txt
python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000
```

---

**Built for the future of artificial intelligence - where machines think, learn, and evolve themselves.**

*© 2024 HRM Project - Autonomous AI Research*