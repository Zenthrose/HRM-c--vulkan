#pragma once

#include <string>
#include <vector>
#include <cstdint>

struct HardwareCapabilities {
    // CPU
    uint32_t cpu_cores;
    uint32_t cpu_threads;
    std::string cpu_architecture;
    uint64_t cpu_cache_size_kb;
    bool has_simd_support;

    // Memory
    uint64_t total_ram_bytes;
    uint64_t available_ram_bytes;
    std::string memory_type; // DDR4, DDR5, LPDDR, etc.

    // GPU
    bool has_gpu;
    bool vulkan_supported;
    uint64_t vram_bytes;
    uint64_t gpu_memory_mb;  // Total GPU memory in MB (derived from vram_bytes)
    bool is_integrated_gpu;  // Whether GPU is integrated (iGPU) vs dedicated (dGPU)
    std::string gpu_name;
    uint32_t gpu_compute_units;

    // Vulkan Capabilities (detailed)
    uint32_t vulkan_api_version;
    uint32_t max_compute_workgroup_size[3];
    uint32_t max_compute_workgroup_invocations;
    uint64_t max_memory_allocation_size;
    uint32_t max_descriptor_sets;
    uint32_t max_bound_descriptor_sets;
    float timestamp_period_ns;
    bool supports_vulkan_1_3;
    bool supports_cooperative_matrix;
    bool supports_subgroup_operations;
    bool supports_raytracing;
    bool supports_mesh_shaders;
    std::vector<std::string> supported_extensions;

    // Storage
    uint64_t total_storage_bytes;
    uint64_t available_storage_bytes;
    std::string storage_type; // SSD, HDD, Flash
    uint32_t storage_speed_mbps;

    // System
    std::string os_name;
    std::string os_version;
    bool is_embedded_system;
    bool has_network_access;

    // Performance tiers
    enum class PerformanceTier {
        ULTRA_LOW,    // <512MB RAM, single core, no GPU
        LOW,          // 512MB-2GB RAM, 1-2 cores
        MEDIUM,       // 2-8GB RAM, 2-4 cores
        HIGH,         // 8-16GB RAM, 4-8 cores
        ULTRA_HIGH    // >16GB RAM, >8 cores
    } performance_tier;
};

class HardwareProfiler {
public:
    HardwareProfiler();
    ~HardwareProfiler();

    HardwareCapabilities profile_system();
    void log_capabilities(const HardwareCapabilities& caps);
    uint64_t get_system_uptime_seconds();

private:
    HardwareCapabilities detect_cpu();
    HardwareCapabilities detect_memory();
    HardwareCapabilities detect_gpu();
    HardwareCapabilities detect_vulkan_capabilities(HardwareCapabilities& caps);
    HardwareCapabilities detect_storage();
    HardwareCapabilities detect_system();
    HardwareCapabilities::PerformanceTier determine_tier(const HardwareCapabilities& caps);
};