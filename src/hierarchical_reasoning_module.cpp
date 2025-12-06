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

    // Input injection (add)
    Tensor current_states = hidden_states;
    // Add input_injection to current_states (with bounds checking)
    if (input_injection.data.size() != current_states.data.size()) {
        std::cerr << "Error: input_injection size mismatch in hierarchical reasoning" << std::endl;
        return current_states;
    }
    for (size_t i = 0; i < current_states.data.size(); ++i) {
        current_states.data[i] += input_injection.data[i];
    }

    // Apply layers with cleanup between passes
    for (size_t i = 0; i < layers_.size(); ++i) {
        current_states = layers_[i]->forward(current_states, cos_sin);
        
        // Force cleanup after each layer to prevent memory accumulation
        if (i % 2 == 1) { // Cleanup every 2 layers
            current_states.data.shrink_to_fit();
        }
    }

    return current_states;
}