#pragma once

#include <vector>
#include <cmath>
#include "attention.hpp" // For Tensor

class LossFunctions {
public:
    // Cross-entropy loss
    static float cross_entropy_loss(const Tensor& logits, const Tensor& targets);

    // Q-learning loss for ACT
    static float q_learning_loss(const Tensor& q_halt_logits, const Tensor& q_continue_logits,
                                const Tensor& target_q_continue, const std::vector<bool>& halted);

    // Compute gradients (simplified)
    static Tensor cross_entropy_gradient(const Tensor& logits, const Tensor& targets);
    static std::pair<Tensor, Tensor> q_learning_gradient(const Tensor& q_halt_logits, const Tensor& q_continue_logits,
                                                        const Tensor& target_q_continue, const std::vector<bool>& halted);

private:
    // Helper functions
    static float softmax_sum(const std::vector<float>& logits);
    static std::vector<float> softmax(const std::vector<float>& logits);
    static std::vector<float> log_softmax(const std::vector<float>& logits);
};