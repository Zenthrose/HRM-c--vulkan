#include "hardware_profiler.hpp"
#include <iostream>
#include <fstream>
#include <thread>
#include <filesystem>
#include <vulkan/vulkan.h>

#ifdef _WIN32
#include <windows.h>
#include <sysinfoapi.h>
#include <psapi.h>
#else
#include <unistd.h>
#include <sys/sysinfo.h>
#endif

namespace fs = std::filesystem;

HardwareProfiler::HardwareProfiler() {
    std::cout << "Hardware Profiler initialized" << std::endl;
}

HardwareProfiler::~HardwareProfiler() {
    std::cout << "Hardware Profiler destroyed" << std::endl;
}

HardwareCapabilities HardwareProfiler::profile_system() {
    HardwareCapabilities caps;

    caps = detect_cpu();
    caps = detect_memory();
    caps = detect_gpu();
    caps = detect_storage();
    caps = detect_system();
    caps.performance_tier = determine_tier(caps);

    log_capabilities(caps);
    return caps;
}

HardwareCapabilities HardwareProfiler::detect_cpu() {
    HardwareCapabilities caps;

    caps.cpu_cores = std::thread::hardware_concurrency();
    caps.cpu_threads = caps.cpu_cores; // Simplified

#ifdef _WIN32
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    caps.cpu_architecture = "x64"; // Simplified
    caps.cpu_cache_size_kb = 256; // Simplified
    caps.has_simd_support = true; // Assume modern CPU
#else
    caps.cpu_architecture = "Unknown";
    caps.cpu_cache_size_kb = 256;
    caps.has_simd_support = true;
#endif

    return caps;
}

HardwareCapabilities HardwareProfiler::detect_memory() {
    HardwareCapabilities caps;

#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    caps.total_ram_bytes = memInfo.ullTotalPhys;
    caps.available_ram_bytes = memInfo.ullAvailPhys;
    caps.memory_type = "DDR4"; // Simplified
#else
    struct sysinfo memInfo;
    sysinfo(&memInfo);
    caps.total_ram_bytes = memInfo.totalram * memInfo.mem_unit;
    caps.available_ram_bytes = memInfo.freeram * memInfo.mem_unit;
    caps.memory_type = "Unknown";
#endif

    return caps;
}

HardwareCapabilities HardwareProfiler::detect_gpu() {
    HardwareCapabilities caps;

    // Vulkan detection
    VkInstance instance;
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
    caps.vulkan_supported = (result == VK_SUCCESS);

    if (caps.vulkan_supported) {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount > 0) {
            caps.has_gpu = true;
            std::vector<VkPhysicalDevice> devices(deviceCount);
            vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(devices[0], &props);
            caps.gpu_name = props.deviceName;

            VkPhysicalDeviceMemoryProperties memProps;
            vkGetPhysicalDeviceMemoryProperties(devices[0], &memProps);

            for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
                if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                    caps.vram_bytes = memProps.memoryHeaps[i].size;
                    break;
                }
            }
        } else {
            caps.has_gpu = false;
            caps.vram_bytes = 0;
            caps.gpu_name = "No Vulkan GPU";
        }
        vkDestroyInstance(instance, nullptr);
    } else {
        caps.has_gpu = false;
        caps.vram_bytes = 0;
        caps.gpu_name = "No Vulkan GPU";
    }

    return caps;
}

HardwareCapabilities HardwareProfiler::detect_storage() {
    HardwareCapabilities caps;

    try {
        fs::space_info space = fs::space("C:/");
        caps.total_storage_bytes = space.capacity;
        caps.available_storage_bytes = space.available;
        caps.storage_type = "SSD"; // Simplified detection
        caps.storage_speed_mbps = 500; // Simplified
    } catch (const std::exception& e) {
        caps.total_storage_bytes = 100000000000; // 100GB default
        caps.available_storage_bytes = 50000000000; // 50GB default
        caps.storage_type = "Unknown";
        caps.storage_speed_mbps = 100;
    }

    return caps;
}

HardwareCapabilities HardwareProfiler::detect_system() {
    HardwareCapabilities caps;

#ifdef _WIN32
    caps.os_name = "Windows";
    caps.os_version = "10/11"; // Simplified
    caps.is_embedded_system = false; // Assume desktop
    caps.has_network_access = true; // Assume connected
#else
    caps.os_name = "Linux/Unix";
    caps.os_version = "Unknown";
    caps.is_embedded_system = false;
    caps.has_network_access = true;
#endif

    return caps;
}

HardwareCapabilities::PerformanceTier HardwareProfiler::determine_tier(const HardwareCapabilities& caps) {
    uint64_t ram_gb = caps.total_ram_bytes / (1024 * 1024 * 1024);
    uint32_t cores = caps.cpu_cores;

    if (ram_gb < 1 || cores < 1) {
        return HardwareCapabilities::PerformanceTier::ULTRA_LOW;
    } else if (ram_gb < 2 || cores < 2) {
        return HardwareCapabilities::PerformanceTier::LOW;
    } else if (ram_gb < 8 || cores < 4) {
        return HardwareCapabilities::PerformanceTier::MEDIUM;
    } else if (ram_gb < 16 || cores < 8) {
        return HardwareCapabilities::PerformanceTier::HIGH;
    } else {
        return HardwareCapabilities::PerformanceTier::ULTRA_HIGH;
    }
}

void HardwareProfiler::log_capabilities(const HardwareCapabilities& caps) {
    std::cout << "=== Hardware Profile ===" << std::endl;
    std::cout << "CPU: " << caps.cpu_cores << " cores, " << caps.cpu_threads << " threads" << std::endl;
    std::cout << "RAM: " << (caps.total_ram_bytes / (1024*1024*1024)) << "GB total, "
              << (caps.available_ram_bytes / (1024*1024*1024)) << "GB available" << std::endl;
    std::cout << "GPU: " << (caps.has_gpu ? "Yes" : "No") << ", Vulkan: " << (caps.vulkan_supported ? "Yes" : "No")
              << ", VRAM: " << (caps.vram_bytes / (1024*1024*1024)) << "GB" << std::endl;
    std::cout << "Storage: " << (caps.total_storage_bytes / (1024LL*1024*1024*1024)) << "TB total" << std::endl;
    std::cout << "Performance Tier: ";

    switch (caps.performance_tier) {
        case HardwareCapabilities::PerformanceTier::ULTRA_LOW: std::cout << "Ultra Low"; break;
        case HardwareCapabilities::PerformanceTier::LOW: std::cout << "Low"; break;
        case HardwareCapabilities::PerformanceTier::MEDIUM: std::cout << "Medium"; break;
        case HardwareCapabilities::PerformanceTier::HIGH: std::cout << "High"; break;
        case HardwareCapabilities::PerformanceTier::ULTRA_HIGH: std::cout << "Ultra High"; break;
    }
    std::cout << std::endl << "========================" << std::endl;
}