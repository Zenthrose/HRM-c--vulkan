#pragma once

#include <vector>
#include <string>

#include "attention.hpp" // For Tensor

struct LinearConfig {
    uint32_t in_features;
    uint32_t out_features;
    bool bias;
};

class LinearVulkan {
public:
    LinearVulkan(const LinearConfig& config, VkPhysicalDevice physicalDevice = VK_NULL_HANDLE, VkDevice device = VK_NULL_HANDLE, VkQueue computeQueue = VK_NULL_HANDLE, uint32_t computeQueueFamilyIndex = 0, VkCommandPool commandPool = VK_NULL_HANDLE);
    ~LinearVulkan();

    Tensor forward(const Tensor& input);

private:
    LinearConfig config;
};