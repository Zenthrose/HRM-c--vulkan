#include <iostream>
#include <vector>
#include "linear.hpp"

LinearVulkan::LinearVulkan(const LinearConfig& config, VkPhysicalDevice physicalDevice, VkDevice device, VkQueue computeQueue, uint32_t computeQueueFamilyIndex, VkCommandPool commandPool)
    : config(config) {
    std::cout << "Initializing LinearVulkan layer..." << std::endl;
    // CPU-only implementation for stability
}

LinearVulkan::~LinearVulkan() {
    std::cout << "Destroying LinearVulkan layer..." << std::endl;
}

Tensor LinearVulkan::forward(const Tensor& input) {
    std::cout << "Performing forward pass in LinearVulkan..." << std::endl;

    // CPU implementation
    size_t batch_size = input.data.size() / config.in_features;
    Tensor output;
    output.shape = {static_cast<unsigned int>(batch_size), config.out_features};
    output.data.resize(batch_size * config.out_features);

    // Simple identity weights for demo
    std::vector<float> weights(config.out_features * config.in_features, 0.0f);
    for (size_t i = 0; i < std::min(static_cast<size_t>(config.in_features), static_cast<size_t>(config.out_features)); ++i) {
        weights[i * config.in_features + i] = 1.0f;
    }

    // Matrix multiplication: output = input @ weights.T
    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t out = 0; out < static_cast<size_t>(config.out_features); ++out) {
            float sum = 0.0f;
            for (size_t in = 0; in < static_cast<size_t>(config.in_features); ++in) {
                sum += input.data[batch * config.in_features + in] * weights[out * config.in_features + in];
            }
            output.data[batch * config.out_features + out] = sum;
        }
    }

    return output;
}