#include "hrm.hpp"
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <tuple>

HRM::HRM(const HRMConfig& config) : config_(config) {
    std::cout << "Initializing HRM..." << std::endl;
    inner_ = std::make_unique<HRMInner>(config.inner_config);
    std::cout << "HRM initialized" << std::endl;
}

HRMCarry HRM::initial_carry(const std::unordered_map<std::string, Tensor>& batch) {
    std::cout << "Creating initial HRM carry..." << std::endl;

    // Get batch size from inputs
    auto inputs_it = batch.find("inputs");
    if (inputs_it == batch.end()) {
        throw std::runtime_error("Missing inputs in batch");
    }
    int batch_size = inputs_it->second.data.size() / config_.inner_config.seq_len; // Approximate

    HRMCarry carry;
    carry.inner_carry = inner_->empty_carry(batch_size);
    carry.steps = std::vector<int32_t>(batch_size, 0);
    carry.halted = std::vector<bool>(batch_size, true); // Default to halted

    // Initialize current_data with empty tensors matching batch
    for (const auto& kv : batch) {
        carry.current_data[kv.first] = kv.second; // Copy for now
    }

    return carry;
}

std::pair<HRMCarry, std::unordered_map<std::string, Tensor>> HRM::forward(
    const HRMCarry& carry,
    const std::unordered_map<std::string, Tensor>& batch
) {
    

    // Update data, carry (removing halted sequences)
    HRMInnerCarry new_inner_carry = inner_->reset_carry(carry.halted, carry.inner_carry);

    std::vector<int32_t> new_steps = carry.steps;
    for (size_t i = 0; i < new_steps.size(); ++i) {
        if (!carry.halted[i]) {
            new_steps[i] += 1;
        } else {
            new_steps[i] = 0;
        }
    }

    std::unordered_map<std::string, Tensor> new_current_data;
    for (const auto& kv : carry.current_data) {
        // Simplified: just copy for now
        new_current_data[kv.first] = kv.second;
    }

    // Forward inner model
    HRMInnerCarry final_inner_carry;
    Tensor logits;
    std::pair<Tensor, Tensor> q_pair;
    std::tie(final_inner_carry, logits, q_pair) = inner_->forward(new_inner_carry, new_current_data);
    Tensor q_halt_logits, q_continue_logits;
    std::tie(q_halt_logits, q_continue_logits) = q_pair;

    // Prepare outputs
    std::unordered_map<std::string, Tensor> outputs;
    outputs["logits"] = logits;
    outputs["q_halt_logits"] = q_halt_logits;
    outputs["q_continue_logits"] = q_continue_logits;

    // Simplified ACT logic (no torch.no_grad for now)
    std::vector<bool> halted = carry.halted;
    bool is_training = config_.is_training;

    for (size_t i = 0; i < new_steps.size(); ++i) {
        bool is_last_step = (new_steps[i] >= config_.inner_config.halt_max_steps);

        if (is_last_step) {
            halted[i] = true;
        } else if (is_training && config_.inner_config.halt_max_steps > 1) {
            // Training logic with exploration
            float halt_prob = 1.0f / (1.0f + std::exp(-q_halt_logits.data[i]));
            float continue_prob = 1.0f / (1.0f + std::exp(-q_continue_logits.data[i]));

            // Simple epsilon-greedy exploration
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);

            if (dist(gen) < config_.inner_config.halt_exploration_prob) {
                // Exploration: random halt step
                int min_halt_steps = 2;
                int max_halt_steps = config_.inner_config.halt_max_steps;
                std::uniform_int_distribution<int> step_dist(min_halt_steps, max_halt_steps);
                int random_halt_step = step_dist(gen);
                halted[i] = (new_steps[i] >= random_halt_step);
            } else {
                // Exploitation: use Q-values
                halted[i] = (halt_prob > continue_prob);
            }
        }
    }

    HRMCarry new_carry{final_inner_carry, new_steps, halted, new_current_data};

    return {new_carry, outputs};
}