#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include "hierarchical_reasoning_module.hpp"
#include "../core/embedding.hpp"
#include "../core/linear.hpp"
#include "../core/attention.hpp" // For Tensor and CosSin

struct HRMInnerConfig {
    int batch_size;
    int seq_len;
    int puzzle_emb_ndim;
    int num_puzzle_identifiers;
    int vocab_size;
    int H_cycles;
    int L_cycles;
    int H_layers;
    int L_layers;
    int hidden_size;
    float expansion;
    int num_heads;
    std::string pos_encodings;
    float rms_norm_eps;
    float rope_theta;
    int halt_max_steps;
    float halt_exploration_prob;
    std::string forward_dtype;

    // Vulkan resources
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue computeQueue;
    uint32_t computeQueueFamilyIndex;
    VkCommandPool commandPool;
};

struct HRMInnerCarry {
    Tensor z_H;
    Tensor z_L;
};

class HRMInner {
public:
    HRMInner(const HRMInnerConfig& config);
    ~HRMInner() = default;

    HRMInnerCarry empty_carry(int batch_size);
    HRMInnerCarry reset_carry(const std::vector<bool>& reset_flag, const HRMInnerCarry& carry);

    std::tuple<HRMInnerCarry, Tensor, std::pair<Tensor, Tensor>> forward(
        const HRMInnerCarry& carry,
        const std::unordered_map<std::string, Tensor>& batch
    );

    std::unordered_map<std::string, Tensor> backward(
        const HRMInnerCarry& carry,
        const std::unordered_map<std::string, Tensor>& batch,
        const std::unordered_map<std::string, Tensor>& output_grads
    );

    // Parameter access for training
    std::unordered_map<std::string, Tensor> get_trainable_parameters();
    void update_parameters(const std::unordered_map<std::string, Tensor>& parameter_updates);

private:
    HRMInnerConfig config_;
    std::unique_ptr<EmbeddingVulkan> embed_tokens_;
    std::unique_ptr<LinearVulkan> lm_head_;
    std::unique_ptr<LinearVulkan> q_head_;
    std::unique_ptr<HierarchicalReasoningModule> H_level_;
    std::unique_ptr<HierarchicalReasoningModule> L_level_;
    Tensor H_init_;
    Tensor L_init_;
    CosSin rotary_emb_;
    float embed_scale_;
    int puzzle_emb_len_;
};