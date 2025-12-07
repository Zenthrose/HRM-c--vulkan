#include "block.hpp"
#include <iostream>

BlockVulkan::BlockVulkan(const BlockConfig& config, VkPhysicalDevice physicalDevice, VkDevice device, VkQueue computeQueue, uint32_t computeQueueFamilyIndex, VkCommandPool commandPool)
    : config(config), physicalDevice(physicalDevice), device(device), computeQueue(computeQueue), computeQueueFamilyIndex(computeQueueFamilyIndex), commandPool(commandPool) {
    std::cout << "Initializing BlockVulkan..." << std::endl;
    attention = std::make_unique<AttentionVulkan>(config.attention_config, physicalDevice, device, computeQueue, computeQueueFamilyIndex, commandPool);
    mlp = std::make_unique<SwiGLUVulkan>(config.mlp_config, physicalDevice, device, computeQueue, computeQueueFamilyIndex, commandPool);
    norm1 = std::make_unique<RMSNormVulkan>(config.norm_config, physicalDevice, device, computeQueue, computeQueueFamilyIndex, commandPool);
    norm2 = std::make_unique<RMSNormVulkan>(config.norm_config, physicalDevice, device, computeQueue, computeQueueFamilyIndex, commandPool);
}

BlockVulkan::~BlockVulkan() {
    std::cout << "Destroying BlockVulkan..." << std::endl;
    // Smart pointers automatically clean up
}

Tensor BlockVulkan::forward(const Tensor& input, const CosSin& cos_sin) {
    
    // Post-norm: norm -> attn -> residual -> norm -> mlp -> residual
    Tensor normed = norm1->forward(input);
    Tensor attn_out = attention->forward(normed, cos_sin);
    Tensor residual1 = input; // Add residual: input + attn_out
    for (size_t i = 0; i < residual1.data.size(); ++i) {
        residual1.data[i] += attn_out.data[i];
    }
    
    Tensor normed2 = norm2->forward(residual1);
    Tensor mlp_out = mlp->forward(normed2);
    Tensor residual2 = residual1; // Add residual: residual1 + mlp_out
    for (size_t i = 0; i < residual2.data.size(); ++i) {
        residual2.data[i] += mlp_out.data[i];
    }
    
    return residual2;
}