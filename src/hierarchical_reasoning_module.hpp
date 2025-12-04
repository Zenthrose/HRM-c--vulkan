#pragma once

#include <vector>
#include <memory>
#include "block.hpp"
#include "attention.hpp" // For Tensor and CosSin

struct HierarchicalReasoningConfig {
    int num_layers;
    BlockConfig block_config;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue computeQueue;
    uint32_t computeQueueFamilyIndex;
    VkCommandPool commandPool;
};

class HierarchicalReasoningModule {
public:
    HierarchicalReasoningModule(const HierarchicalReasoningConfig& config);
    ~HierarchicalReasoningModule() = default;

    Tensor forward(const Tensor& hidden_states, const Tensor& input_injection, const CosSin& cos_sin);

private:
    std::vector<std::unique_ptr<BlockVulkan>> layers_;
};