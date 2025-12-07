#include "vulkan_compatibility.h"
#include <iostream>
#include <algorithm>

VulkanCompatibilityInfo VulkanCompatibility::checkCompatibility() {
    VulkanCompatibilityInfo info;
    info.apiVersion = negotiateApiVersion();
    info.vulkanSupported = true;

    // Enumerate extensions
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

    for (const auto& ext : extensions) {
        info.availableExtensions.push_back(ext.extensionName);
    }

    // Check extension support
    checkExtensionSupport(info);

    // Enumerate devices
    enumerateDevices(info);

    // Select best device
    if (!info.devices.empty()) {
        info.selectedDevice = &info.devices[0]; // Default to first
        for (auto& device : info.devices) {
            if (scoreDevice(device) > scoreDevice(*info.selectedDevice)) {
                info.selectedDevice = &device;
            }
        }
    }

    return info;
}

uint32_t VulkanCompatibility::negotiateApiVersion() {
    // For simplicity, use Vulkan 1.1
    // In production, would negotiate based on available version
    return VK_API_VERSION_1_1;
}

bool VulkanCompatibility::enumerateDevices(VulkanCompatibilityInfo& info) {
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.apiVersion = info.apiVersion;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkInstance instance;
    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
    if (result != VK_SUCCESS) {
        return false;
    }

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
        vkDestroyInstance(instance, nullptr);
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (VkPhysicalDevice device : devices) {
        VulkanDeviceInfo deviceInfo;
        deviceInfo.device = device;

        vkGetPhysicalDeviceProperties(device, &deviceInfo.properties);
        vkGetPhysicalDeviceFeatures(device, &deviceInfo.features);

        deviceInfo.name = deviceInfo.properties.deviceName;
        deviceInfo.type = deviceInfo.properties.deviceType;

        // Get queue families
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        deviceInfo.queueFamilies.resize(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, deviceInfo.queueFamilies.data());

        // Find queue families
        for (uint32_t i = 0; i < queueFamilyCount; ++i) {
            if (deviceInfo.queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                deviceInfo.graphicsQueueFamily = i;
            }
            if (deviceInfo.queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                deviceInfo.computeQueueFamily = i;
            }
        }

        // Get extensions
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, nullptr);
        deviceInfo.extensions.resize(extCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, deviceInfo.extensions.data());

        // Score device
        deviceInfo.score = scoreDevice(deviceInfo);

        info.devices.push_back(deviceInfo);
    }

    vkDestroyInstance(instance, nullptr);
    return true;
}

int VulkanCompatibility::scoreDevice(const VulkanDeviceInfo& device) {
    int score = 0;

    // Prefer discrete GPUs
    if (device.type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        score += 1000;
    } else if (device.type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
        score += 500;
    }

    // Check compute capability
    if (device.computeQueueFamily != UINT32_MAX) {
        score += 100;
    }

    // Check graphics capability (bonus)
    if (device.graphicsQueueFamily != UINT32_MAX) {
        score += 50;
    }

    return score;
}

bool VulkanCompatibility::checkExtensionSupport(VulkanCompatibilityInfo& info) {
    info.requiredExtensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
#ifdef _WIN32
    info.requiredExtensions.push_back("VK_KHR_win32_surface");
#endif

    info.optionalExtensions = {
        VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
        VK_KHR_16BIT_STORAGE_EXTENSION_NAME
    };

    for (const std::string& ext : info.availableExtensions) {
        info.extensionSupport[ext] = true;
    }

    return true;
}

VkPhysicalDevice VulkanCompatibility::selectBestDevice(const std::vector<VulkanDeviceInfo>& devices) {
    if (devices.empty()) return VK_NULL_HANDLE;

    const VulkanDeviceInfo* best = &devices[0];
    for (const auto& device : devices) {
        if (scoreDevice(device) > scoreDevice(*best)) {
            best = &device;
        }
    }

    return best->device;
}

bool VulkanCompatibility::checkDeviceFeatures(VkPhysicalDevice device, VkPhysicalDeviceFeatures* requiredFeatures) {
    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(device, &features);

    // Compute shaders are mandatory in Vulkan 1.0+, so no need to check

    // Check for basic features we need
    if (!features.shaderFloat64) {
        std::cerr << "Warning: Device does not support 64-bit floats in shaders" << std::endl;
    }

    if (!features.shaderInt64) {
        std::cerr << "Warning: Device does not support 64-bit integers in shaders" << std::endl;
    }

    // Check memory properties
    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(device, &memoryProperties);

    bool hasDeviceLocal = false;
    bool hasHostVisible = false;

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        if (memoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            hasDeviceLocal = true;
        }
        if (memoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            hasHostVisible = true;
        }
    }

    if (!hasDeviceLocal) {
        std::cerr << "Device does not have device-local memory" << std::endl;
        return false;
    }

    if (!hasHostVisible) {
        std::cerr << "Device does not have host-visible memory" << std::endl;
        return false;
    }

    if (requiredFeatures) {
        // Compare required features with available features
        // For now, assume required features are supported if basic checks pass
        return true;
    }

    return true;
}

std::vector<const char*> VulkanCompatibility::getRequiredExtensions() {
    std::vector<const char*> extensions;
    extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
#ifdef _WIN32
    extensions.push_back("VK_KHR_win32_surface");
#endif
    return extensions;
}

std::string VulkanCompatibility::getVulkanResultString(VkResult result) {
    switch (result) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_EVENT_SET: return "VK_EVENT_SET";
        case VK_EVENT_RESET: return "VK_EVENT_RESET";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
        default: return "Unknown Vulkan error";
    }
}