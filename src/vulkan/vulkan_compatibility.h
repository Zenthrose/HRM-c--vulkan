#pragma once

#ifndef NO_VULKAN
#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <unordered_map>

struct VulkanDeviceInfo {
    VkPhysicalDevice device;
    VkPhysicalDeviceProperties properties;
    VkPhysicalDeviceFeatures features;
    std::vector<VkExtensionProperties> extensions;
    std::vector<VkQueueFamilyProperties> queueFamilies;
    uint32_t graphicsQueueFamily = UINT32_MAX;
    uint32_t computeQueueFamily = UINT32_MAX;
    int score = 0;
    std::string name;
    VkPhysicalDeviceType type;
};

struct VulkanCompatibilityInfo {
    bool vulkanSupported = false;
    uint32_t apiVersion = VK_API_VERSION_1_0;
    VkInstance instance = VK_NULL_HANDLE;
    std::vector<VulkanDeviceInfo> devices;
    VulkanDeviceInfo* selectedDevice = nullptr;
    std::vector<std::string> availableExtensions;
    std::vector<std::string> requiredExtensions;
    std::vector<std::string> optionalExtensions;
    std::unordered_map<std::string, bool> extensionSupport;
};

class VulkanCompatibility {
public:
    static VulkanCompatibilityInfo checkCompatibility();
    static VkPhysicalDevice selectBestDevice(const std::vector<VulkanDeviceInfo>& devices);
    static bool testDeviceBasicFeatures(VkPhysicalDevice device);
    static bool checkDeviceFeatures(VkPhysicalDevice device, VkPhysicalDeviceFeatures* requiredFeatures = nullptr);
    static std::vector<const char*> getRequiredExtensions();
    static uint32_t negotiateApiVersion();
    static std::string getVulkanResultString(VkResult result);

private:
    static int scoreDevice(const VulkanDeviceInfo& device);
    static bool enumerateDevices(VulkanCompatibilityInfo& info);
    static bool checkExtensionSupport(VulkanCompatibilityInfo& info);
};
#endif