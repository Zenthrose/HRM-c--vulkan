#pragma once

#ifndef NO_VULKAN
#include <vulkan/vulkan.h>
#endif
#include <vector>
#include <string>
#include <memory>

#include "attention.hpp" // For Tensor
#include "linear.hpp" // For LinearVulkan

struct SwiGLUConfig {
    uint32_t hidden_size;
    float expansion;
};

#ifndef NO_VULKAN
class SwiGLUVulkan {
public:
    SwiGLUVulkan(const SwiGLUConfig& config, VkPhysicalDevice physicalDevice, VkDevice device, VkQueue computeQueue, uint32_t computeQueueFamilyIndex, VkCommandPool commandPool);
    ~SwiGLUVulkan();

    Tensor forward(const Tensor& input);

private:
    SwiGLUConfig config;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue computeQueue;
    uint32_t computeQueueFamilyIndex;
    VkCommandPool commandPool;

    std::unique_ptr<LinearVulkan> gate_up_proj;
    std::unique_ptr<LinearVulkan> down_proj;
};
#endif
