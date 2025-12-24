#pragma once

#include <vector>
#include <memory>
#include "../core/block.hpp"
#include "../core/attention.hpp" // For Tensor and CosSin

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
    
    // Access individual layers for HRM inner forward (matching original Python design)
    BlockVulkan* get_layer(size_t index) { return layers_[index].get(); }
    size_t get_num_layers() const { return layers_.size(); }
 
private:
    std::vector<std::unique_ptr<BlockVulkan>> layers_;
};