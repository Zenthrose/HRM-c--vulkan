# Changelog

All notable changes to the HRM (Hierarchical Reasoning Model) project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Portability improvements across OS (Windows/Linux/macOS)
- Environment variable support for config, logs, models, compactions
- Docker multi-stage build support
- Restructured source code into subdirectories for better organization
- CMake improvements with conditional DLL copying and portable paths
- Enhanced .gitignore for cross-platform development

### Changed
- Updated build system for universal compatibility
- Improved configuration management with env vars
- Restructured src/ directory with core, hrm, vulkan, self_mod, system, training, utils, main subdirs

### Fixed
- Hard-coded paths replaced with portable alternatives
- DLL copying now uses find_program for MinGW detection
- Temp directories use std::filesystem::temp_directory_path()

## [1.0.0] - 2024-12-XX

### Added
- Initial release of HRM with Vulkan acceleration
- Self-modifying, self-repairing, resource-aware AI
- Character-level language processing
- Conversational AI capabilities
- Cross-platform build support
- Advanced training optimizations (FlashAttention, Mixed Precision)
- Cloud storage integration
- GUI and CLI interfaces

### Features
- Hierarchical reasoning with adaptive computation time
- Vulkan-based neural network training
- Self-evolution framework
- Resource intelligence and OOM prevention
- UTF-8 processing for multilingual support
- Memory compaction and distributed storage