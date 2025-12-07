#include "hrm_inner.hpp"
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>

// Helper function for tensor addition with memory optimization
Tensor tensor_add(const Tensor& a, const Tensor& b) {
    if (a.data.size() != b.data.size()) {
        throw std::runtime_error("Tensor size mismatch in addition");
    }
    Tensor result;
    result.data.reserve(a.data.size()); // Reserve exact size to avoid reallocations
    result.data.resize(a.data.size());
    for (size_t i = 0; i < a.data.size(); ++i) {
        result.data[i] = a.data[i] + b.data[i];
    }
    result.data.shrink_to_fit(); // Free any excess capacity
    return result;
}

// Helper function to create tensor with specific size and memory optimization
Tensor create_tensor(size_t size, float value) {
    Tensor result;
    result.data.reserve(size); // Reserve exact size
    result.data.resize(size, value);
    result.data.shrink_to_fit(); // Ensure no excess capacity
    return result;
}

HRMInner::HRMInner(const HRMInnerConfig& config) : config_(config) {
    std::cout << "Initializing HRMInner..." << std::endl;

    embed_scale_ = std::sqrt(config.hidden_size);
    puzzle_emb_len_ = (config.puzzle_emb_ndim + config.hidden_size - 1) / config.hidden_size; // ceil div

    // Initialize embeddings
    EmbeddingConfig embed_config{static_cast<uint32_t>(config.vocab_size), static_cast<uint32_t>(config.hidden_size), static_cast<uint32_t>(config.seq_len)};
    embed_tokens_ = std::make_unique<EmbeddingVulkan>(
        embed_config, config.physicalDevice, config.device, config.computeQueue,
        config.computeQueueFamilyIndex, config.commandPool
    );

    // Initialize LM head
    LinearConfig lm_config{static_cast<uint32_t>(config.hidden_size), static_cast<uint32_t>(config.vocab_size), false};
    lm_head_ = std::make_unique<LinearVulkan>(
        lm_config, config.physicalDevice, config.device, config.computeQueue,
        config.computeQueueFamilyIndex, config.commandPool
    );

    // Initialize Q head
    LinearConfig q_config{static_cast<uint32_t>(config.hidden_size), 2, true};
    q_head_ = std::make_unique<LinearVulkan>(
        q_config, config.physicalDevice, config.device, config.computeQueue,
        config.computeQueueFamilyIndex, config.commandPool
    );

    // Initialize reasoning modules
    BlockConfig block_config{
        AttentionConfig{
            static_cast<uint32_t>(config.batch_size),
            static_cast<uint32_t>(config.seq_len + puzzle_emb_len_),
            static_cast<uint32_t>(config.hidden_size / config.num_heads),
            static_cast<uint32_t>(config.num_heads),
            static_cast<uint32_t>(config.num_heads),
            false
        },
        SwiGLUConfig{
            static_cast<uint32_t>(config.hidden_size),
            static_cast<float>(config.expansion)
        },
        RMSNormConfig{
            static_cast<uint32_t>(config.seq_len + puzzle_emb_len_),
            static_cast<uint32_t>(config.hidden_size),
            config.rms_norm_eps
        }
    };

    HierarchicalReasoningConfig h_level_config{
        config.H_layers,
        block_config,
        config.physicalDevice, config.device, config.computeQueue,
        config.computeQueueFamilyIndex, config.commandPool
    };
    H_level_ = std::make_unique<HierarchicalReasoningModule>(h_level_config);

    HierarchicalReasoningConfig l_level_config{
        config.L_layers,
        block_config,
        config.physicalDevice, config.device, config.computeQueue,
        config.computeQueueFamilyIndex, config.commandPool
    };
    L_level_ = std::make_unique<HierarchicalReasoningModule>(l_level_config);

    // Initialize rotary embeddings if needed
    if (config.pos_encodings == "rope") {
        // Implement RoPE (Rotary Position Embedding)
        int max_seq_len = config.seq_len;
        int head_dim = config.hidden_size / config.num_heads;

        std::vector<float> cos_emb(max_seq_len * head_dim / 2);
        std::vector<float> sin_emb(max_seq_len * head_dim / 2);

        for (int pos = 0; pos < max_seq_len; ++pos) {
            for (int i = 0; i < head_dim / 2; ++i) {
                float theta = std::pow(10000.0f, -2.0f * i / head_dim);
                float angle = pos * theta;
                cos_emb[pos * (head_dim / 2) + i] = std::cos(angle);
                sin_emb[pos * (head_dim / 2) + i] = std::sin(angle);
            }
        }

        rotary_emb_ = CosSin{cos_emb, sin_emb};
    }

    // Initialize H_init and L_init with truncated normal
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    H_init_.data.resize(config.hidden_size);
    L_init_.data.resize(config.hidden_size);
    for (auto& val : H_init_.data) val = dist(gen);
    for (auto& val : L_init_.data) val = dist(gen);

    std::cout << "HRMInner initialized" << std::endl;
}

HRMInnerCarry HRMInner::empty_carry(int batch_size) {
    // Simplified for character training: ignore sequence and puzzle lengths
    int tensor_size = batch_size * config_.hidden_size;

    HRMInnerCarry carry;
    carry.z_H.data.resize(tensor_size, 0.0f);
    carry.z_H.shape = {static_cast<uint32_t>(batch_size), static_cast<uint32_t>(config_.hidden_size)};
    carry.z_L.data.resize(tensor_size, 0.0f);
    carry.z_L.shape = {static_cast<uint32_t>(batch_size), static_cast<uint32_t>(config_.hidden_size)};

    return carry;
}

HRMInnerCarry HRMInner::reset_carry(const std::vector<bool>& reset_flag, const HRMInnerCarry& carry) {
    HRMInnerCarry new_carry = carry;

    for (size_t b = 0; b < reset_flag.size(); ++b) {
        if (reset_flag[b]) {
            // Reset to H_init and L_init
            size_t offset = b * config_.hidden_size;
            for (int i = 0; i < config_.hidden_size; ++i) {
                new_carry.z_H.data[offset + i] = H_init_.data[i];
                new_carry.z_L.data[offset + i] = L_init_.data[i];
            }
        }
    }

    return new_carry;
}

std::tuple<HRMInnerCarry, Tensor, std::pair<Tensor, Tensor>> HRMInner::forward(
    const HRMInnerCarry& carry,
    const std::unordered_map<std::string, Tensor>& batch
) {
    

    // Get inputs from batch
    auto inputs_it = batch.find("inputs");
    auto puzzle_identifiers_it = batch.find("puzzle_identifiers");
    if (inputs_it == batch.end() || puzzle_identifiers_it == batch.end()) {
        throw std::runtime_error("Missing inputs or puzzle_identifiers in batch");
    }

    // For now, simplified implementation - need to implement full input embeddings
    Tensor input_embeddings = embed_tokens_->forward(std::vector<uint32_t>(inputs_it->second.data.begin(), inputs_it->second.data.end()));

    // Scale embeddings
    for (auto& val : input_embeddings.data) {
        val *= embed_scale_;
    }

    // Forward iterations - matching original HRM design exactly
    Tensor z_H = carry.z_H;
    Tensor z_L = carry.z_L;

    for (int H_step = 0; H_step < config_.H_cycles; ++H_step) {
        for (int L_step = 0; L_step < config_.L_cycles; ++L_step) {
            if (!(H_step == config_.H_cycles - 1 && L_step == config_.L_cycles - 1)) {
                // L_level forward - apply individual layers directly (matching original Python design)
                Tensor L_input = tensor_add(z_H, input_embeddings);
                
                // Apply L layers sequentially like original Python code
                for (size_t i = 0; i < config_.L_layers; ++i) {
                    L_input = L_level_->get_layer(i)->forward(L_input, rotary_emb_);
                }
                z_L = L_input;
            }
        }

        if (H_step != config_.H_cycles - 1) {
            // H_level forward - apply individual layers directly (matching original Python design)
            Tensor H_input = z_L;
            
            // Apply H layers sequentially like original Python code
            for (size_t i = 0; i < config_.H_layers; ++i) {
                H_input = H_level_->get_layer(i)->forward(H_input, rotary_emb_);
            }
            z_H = H_input;
        }
    }
    
    

    // 1-step grad
    z_L = L_level_->forward(z_L, tensor_add(z_H, input_embeddings), rotary_emb_);
    z_H = H_level_->forward(z_H, z_L, rotary_emb_);

    // LM Outputs
    HRMInnerCarry new_carry{z_H, z_L}; // No detach for now
    Tensor logits = lm_head_->forward(z_H);

    // Q head (simplified)
    Tensor q_logits = q_head_->forward(z_H);

    // Extract q_halt and q_continue from q_logits
    Tensor q_halt = create_tensor(q_logits.data.size() / 2, 0.0f);
    Tensor q_continue = create_tensor(q_logits.data.size() / 2, 0.0f);
    for (size_t i = 0; i < q_halt.data.size(); ++i) {
        q_halt.data[i] = q_logits.data[i * 2];
        q_continue.data[i] = q_logits.data[i * 2 + 1];
    }

    return {new_carry, logits, {q_halt, q_continue}};
}