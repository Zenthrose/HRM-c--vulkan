#include "hardware_profiler.hpp"
#include <iostream>
#include <fstream>
#include <thread>
#include <filesystem>
#include <cstring>
#ifndef NO_VULKAN
#include <vulkan/vulkan.h>
#endif
#include "../vulkan/vulkan_instance_manager.hpp"

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
    if (caps.vulkan_supported) {
        caps = detect_vulkan_capabilities(caps);
    }
    caps = detect_storage();
    caps = detect_system();
    // Calculate GPU memory in MB for easier tier determination
    caps.gpu_memory_mb = caps.vram_bytes / (1024 * 1024);

    // Determine if GPU is integrated vs dedicated
    // Intel GPUs are typically integrated, AMD/NVIDIA are dedicated
    caps.is_integrated_gpu = caps.gpu_name.find("Intel") != std::string::npos ||
                            caps.gpu_name.find("UHD") != std::string::npos ||
                            caps.gpu_name.find("Iris") != std::string::npos;

    caps.performance_tier = determine_tier(caps);

    log_capabilities(caps);
    return caps;
}

uint64_t HardwareProfiler::get_system_uptime_seconds() {
#ifdef _WIN32
    return GetTickCount64() / 1000; // Windows uptime in seconds
#else
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.uptime; // Linux uptime in seconds
    }
    return 0; // Fallback
#endif
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
            std::vector<VkPhysicalDevice> devices(deviceCount);
            VkResult enumResult = vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

            if (enumResult == VK_SUCCESS && deviceCount > 0 && devices[0] != VK_NULL_HANDLE) {
                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(devices[0], &props);

            caps.gpu_name = props.deviceName;

            VkPhysicalDeviceMemoryProperties memProps;
            vkGetPhysicalDeviceMemoryProperties(devices[0], &memProps);

            uint64_t totalVram = 0;
            for (uint32_t i = 0; i < memProps.memoryHeapCount; i++) {
                if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                    totalVram += memProps.memoryHeaps[i].size;
                }
            }
                caps.vram_bytes = totalVram;

                vkDestroyInstance(instance, nullptr);
            } else {
                // Enumeration failed or no valid devices
                caps.vulkan_supported = false;
                caps.vram_bytes = 0;
                caps.gpu_compute_units = 0;
                caps.gpu_name = "Vulkan enumeration failed";
                vkDestroyInstance(instance, nullptr);
            }
        }
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
    uint64_t gpu_mb = caps.gpu_memory_mb;
    bool is_integrated = caps.is_integrated_gpu;

    // Determine base tier from CPU/RAM (existing logic)
    HardwareCapabilities::PerformanceTier base_tier;
    if (ram_gb < 1 || cores < 1) {
        base_tier = HardwareCapabilities::PerformanceTier::ULTRA_LOW;
    } else if (ram_gb < 2 || cores < 2) {
        base_tier = HardwareCapabilities::PerformanceTier::LOW;
    } else if (ram_gb < 8 || cores < 4) {
        base_tier = HardwareCapabilities::PerformanceTier::MEDIUM;
    } else if (ram_gb < 16 || cores < 8) {
        base_tier = HardwareCapabilities::PerformanceTier::HIGH;
    } else {
        base_tier = HardwareCapabilities::PerformanceTier::ULTRA_HIGH;
    }

    // Apply intelligent GPU-aware constraints
    // Only downgrade when GPU is genuinely incapable (integrated GPUs with low memory)

    if (is_integrated && gpu_mb < 4096) {
        // Integrated GPUs with < 4GB need downgrade (like Intel Iris Xe)
        if (gpu_mb < 2048) {
            return HardwareCapabilities::PerformanceTier::LOW;
        } else {
            return HardwareCapabilities::PerformanceTier::MEDIUM;
        }
    } else if (!is_integrated && gpu_mb >= 4096) {
        // Dedicated GPUs with >= 4GB can use full capabilities (like AMD RX 580)
        return base_tier;
    } else if (gpu_mb < 1024) {
        // Any GPU with < 1GB is ultra-low
        return HardwareCapabilities::PerformanceTier::ULTRA_LOW;
    }

    // Default: Use base tier (CPU/RAM determined)
    // This preserves high-end capabilities for capable systems
    return base_tier;
}

HardwareCapabilities HardwareProfiler::detect_vulkan_capabilities(HardwareCapabilities& caps) {
    // COMMENTED OUT: Direct Vulkan instance creation - now use shared instance
    /*
    // Create Vulkan instance to query capabilities
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.apiVersion = VK_API_VERSION_1_3;  // Request Vulkan 1.3

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkInstance instance;
    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan instance for capability detection" << std::endl;
        return caps;
    }
    */

    // Use shared Vulkan instance with ownership validation
    VulkanInstanceManager& manager = VulkanInstanceManager::getInstance();

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

    if (manager.isSharedInstanceAvailable()) {
        instance = manager.getSharedInstance();
        physicalDevice = manager.getSharedPhysicalDevice();
        std::cout << "HardwareProfiler: Using shared Vulkan instance for capability detection" << std::endl;
    } else {
        std::cout << "HardwareProfiler: Shared Vulkan instance not available, creating fallback" << std::endl;
        std::cout << "HardwareProfiler: DEBUG - This should not happen if VulkanCompatibility registered the instance" << std::endl;
        // Create fallback instance for capability detection
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
        if (result != VK_SUCCESS) {
            std::cerr << "HardwareProfiler: Failed to create fallback Vulkan instance" << std::endl;
            return caps;
        }
    }

    // Perform capability detection using instance
    if (instance != VK_NULL_HANDLE) {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount > 0) {
            // Use provided physical device or get first available
            if (physicalDevice == VK_NULL_HANDLE) {
                std::vector<VkPhysicalDevice> devices(deviceCount);
                vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
                if (!devices.empty() && devices[0] != VK_NULL_HANDLE) {
                    physicalDevice = devices[0];
                }
            }

            if (physicalDevice != VK_NULL_HANDLE) {
                // Get API version
                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(physicalDevice, &props);
                caps.vulkan_api_version = props.apiVersion;

                // Get compute limits
                VkPhysicalDeviceLimits limits = props.limits;
                caps.max_compute_workgroup_size[0] = limits.maxComputeWorkGroupSize[0];
                caps.max_compute_workgroup_size[1] = limits.maxComputeWorkGroupSize[1];
                caps.max_compute_workgroup_size[2] = limits.maxComputeWorkGroupSize[2];
                caps.max_compute_workgroup_invocations = limits.maxComputeWorkGroupInvocations;
                caps.max_memory_allocation_size = limits.maxUniformBufferRange;
                caps.timestamp_period_ns = limits.timestampPeriod;

                // Get descriptor limits
                caps.max_descriptor_sets = limits.maxBoundDescriptorSets;
                caps.max_bound_descriptor_sets = limits.maxBoundDescriptorSets;

                // Check Vulkan 1.3 support
                caps.supports_vulkan_1_3 = (VK_API_VERSION_MAJOR(caps.vulkan_api_version) >= 1 &&
                                           VK_API_VERSION_MINOR(caps.vulkan_api_version) >= 3);

                // Query extensions
                uint32_t extensionCount = 0;
                vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);

                std::vector<VkExtensionProperties> extensions(extensionCount);
                vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, extensions.data());

                for (const auto& ext : extensions) {
                    caps.supported_extensions.push_back(ext.extensionName);

                    // Check for advanced features
                    if (strcmp(ext.extensionName, "VK_KHR_cooperative_matrix") == 0) {
                        caps.supports_cooperative_matrix = true;
                    }
                    if (strcmp(ext.extensionName, "VK_EXT_shader_subgroup") == 0) {
                        caps.supports_subgroup_operations = true;
                    }
                    if (strcmp(ext.extensionName, "VK_KHR_ray_tracing_pipeline") == 0) {
                        caps.supports_raytracing = true;
                    }
                    if (strcmp(ext.extensionName, "VK_EXT_mesh_shader") == 0) {
                        caps.supports_mesh_shaders = true;
                    }
                }

                // Query Vulkan 1.3 features if supported
                if (caps.supports_vulkan_1_3) {
                    VkPhysicalDeviceVulkan13Features vulkan13Features{};
                    vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

                    VkPhysicalDeviceFeatures2 features2{};
                    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
                    features2.pNext = &vulkan13Features;

                    vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);
                }
            }
        }
    }

    // CRITICAL: Validate instance destruction permissions (ALWAYS ACTIVE)
    manager.validateInstanceDestruction(instance);

    // COMMENTED OUT: Only destroy if it's a fallback instance (not shared)
    /*
    vkDestroyInstance(instance, nullptr);
    */
    // Fallback instances can be destroyed, shared instances are managed by VulkanInstanceManager

    if (!manager.isSharedInstance(instance) && instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
    }

    return caps;
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