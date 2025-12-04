#include "hierarchical_reasoning_module.hpp"
#include <iostream>

HierarchicalReasoningModule::HierarchicalReasoningModule(const HierarchicalReasoningConfig& config) {
    std::cout << "Initializing HierarchicalReasoningModule with " << config.num_layers << " layers..." << std::endl;

    for (int i = 0; i < config.num_layers; ++i) {
        layers_.push_back(std::make_unique<BlockVulkan>(
            config.block_config,
            config.physicalDevice,
            config.device,
            config.computeQueue,
            config.computeQueueFamilyIndex,
            config.commandPool
        ));
    }

    std::cout << "Initialized HierarchicalReasoningModule with " << layers_.size() << " layers" << std::endl;
}

Tensor HierarchicalReasoningModule::forward(const Tensor& hidden_states, const Tensor& input_injection, const CosSin& cos_sin) {
    std::cout << "HierarchicalReasoningModule forward..." << std::endl;

    // Input injection (add)
    Tensor current_states = hidden_states;
    // Add input_injection to current_states
    for (size_t i = 0; i < current_states.data.size(); ++i) {
        current_states.data[i] += input_injection.data[i];
    }

    // Apply layers
    for (auto& layer : layers_) {
        current_states = layer->forward(current_states, cos_sin);
    }

    return current_states;
}