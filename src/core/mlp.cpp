#include "mlp.hpp"
#include "linear.hpp"
#include <iostream>
#include <cmath>
#include <memory>

SwiGLUVulkan::SwiGLUVulkan(const SwiGLUConfig& config, VkPhysicalDevice physicalDevice, VkDevice device, VkQueue computeQueue, uint32_t computeQueueFamilyIndex, VkCommandPool commandPool)
    : config(config), physicalDevice(physicalDevice), device(device), computeQueue(computeQueue), computeQueueFamilyIndex(computeQueueFamilyIndex), commandPool(commandPool) {
    std::cout << "Initializing SwiGLUVulkan layer..." << std::endl;

    // Calculate intermediate size: ceil(expansion * hidden_size * 2 / 3)
    uint32_t inter = static_cast<uint32_t>(std::ceil(config.expansion * config.hidden_size * 2.0f / 3.0f));

    // Create gate_up projection: hidden_size -> inter * 2 (gate + up)
    LinearConfig gate_up_config = {config.hidden_size, inter * 2, false}; // No bias
    gate_up_proj = std::make_unique<LinearVulkan>(gate_up_config, physicalDevice, device, computeQueue, computeQueueFamilyIndex, commandPool);

    // Create down projection: inter -> hidden_size
    LinearConfig down_config = {inter, config.hidden_size, false}; // No bias
    down_proj = std::make_unique<LinearVulkan>(down_config, physicalDevice, device, computeQueue, computeQueueFamilyIndex, commandPool);
}

SwiGLUVulkan::~SwiGLUVulkan() {
    std::cout << "Destroying SwiGLUVulkan layer..." << std::endl;
    // Smart pointers automatically clean up
}

Tensor SwiGLUVulkan::forward(const Tensor& input) {

    // 1. Gate-up projection: input -> [gate, up]
    Tensor gate_up = gate_up_proj->forward(input);

    // 2. Split into gate and up
    uint32_t inter = gate_up.data.size() / 2;
    Tensor gate, up;
    gate.shape = input.shape; // Same shape as input
    up.shape = input.shape;
    gate.data.resize(inter);
    up.data.resize(inter);

    for (size_t i = 0; i < inter; ++i) {
        gate.data[i] = gate_up.data[i];
        up.data[i] = gate_up.data[i + inter];
    }

    // 3. Apply SiLU to gate: gate = gate * sigmoid(gate)
    for (size_t i = 0; i < gate.data.size(); ++i) {
        float x = gate.data[i];
        gate.data[i] = x * (1.0f / (1.0f + std::exp(-x))); // SiLU
    }

    // 4. Element-wise multiply: gate * up
    Tensor gate_up_product;
    gate_up_product.shape = gate.shape;
    gate_up_product.data.resize(gate.data.size());
    for (size_t i = 0; i < gate.data.size(); ++i) {
        gate_up_product.data[i] = gate.data[i] * up.data[i];
    }

    // 5. Down projection: (gate * up) -> output
    Tensor output = down_proj->forward(gate_up_product);

    return output;
}