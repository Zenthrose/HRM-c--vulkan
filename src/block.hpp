#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <memory>

#include "attention.hpp"
#include "mlp.hpp"
#include "rms_norm.hpp"

struct BlockConfig {
    AttentionConfig attention_config;
    SwiGLUConfig mlp_config;
    RMSNormConfig norm_config;
};

class BlockVulkan {
public:
    BlockVulkan(const BlockConfig& config, VkPhysicalDevice physicalDevice, VkDevice device, VkQueue computeQueue, uint32_t computeQueueFamilyIndex, VkCommandPool commandPool);
    ~BlockVulkan();

    Tensor forward(const Tensor& input, const CosSin& cos_sin);

private:
    BlockConfig config;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue computeQueue;
    uint32_t computeQueueFamilyIndex;
    VkCommandPool commandPool;

    std::unique_ptr<AttentionVulkan> attention;
    std::unique_ptr<SwiGLUVulkan> mlp;
    std::unique_ptr<RMSNormVulkan> norm1;
    std::unique_ptr<RMSNormVulkan> norm2;
};