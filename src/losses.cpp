#include "losses.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>

// Cross-entropy loss
float LossFunctions::cross_entropy_loss(const Tensor& logits, const Tensor& targets) {
    // Simplified: assume logits and targets are 1D vectors of same size
    if (logits.data.size() != targets.data.size()) {
        throw std::runtime_error("Logits and targets size mismatch");
    }

    std::vector<float> log_probs = log_softmax(logits.data);
    float loss = 0.0f;

    for (size_t i = 0; i < targets.data.size(); ++i) {
        int target_idx = static_cast<int>(targets.data[i]);
        if (target_idx >= 0 && target_idx < static_cast<int>(logits.data.size())) {
            loss -= log_probs[target_idx];
        }
    }

    return loss / targets.data.size();
}

// Q-learning loss for ACT
float LossFunctions::q_learning_loss(const Tensor& q_halt_logits, const Tensor& q_continue_logits,
                                   const Tensor& target_q_continue, const std::vector<bool>& halted) {
    // Simplified Q-learning loss
    float loss = 0.0f;
    size_t count = 0;

    for (size_t i = 0; i < q_halt_logits.data.size(); ++i) {
        if (!halted[i]) {
            // TD error for continue action
            float q_continue = 1.0f / (1.0f + std::exp(-q_continue_logits.data[i]));
            float td_error = q_continue - target_q_continue.data[i];
            loss += td_error * td_error;
            count++;
        }
    }

    return count > 0 ? loss / count : 0.0f;
}

// Compute cross-entropy gradients
Tensor LossFunctions::cross_entropy_gradient(const Tensor& logits, const Tensor& targets) {
    Tensor grad;
    grad.data = softmax(logits.data);

    for (size_t i = 0; i < targets.data.size(); ++i) {
        int target_idx = static_cast<int>(targets.data[i]);
        if (target_idx >= 0 && target_idx < static_cast<int>(grad.data.size())) {
            grad.data[target_idx] -= 1.0f;
        }
    }

    // Normalize by batch size
    for (auto& val : grad.data) {
        val /= targets.data.size();
    }

    return grad;
}

// Compute Q-learning gradients
std::pair<Tensor, Tensor> LossFunctions::q_learning_gradient(const Tensor& q_halt_logits, const Tensor& q_continue_logits,
                                                           const Tensor& target_q_continue, const std::vector<bool>& halted) {
    Tensor grad_halt, grad_continue;
    grad_halt.data.resize(q_halt_logits.data.size(), 0.0f);
    grad_continue.data.resize(q_continue_logits.data.size(), 0.0f);

    for (size_t i = 0; i < q_continue_logits.data.size(); ++i) {
        if (!halted[i]) {
            float q_continue = 1.0f / (1.0f + std::exp(-q_continue_logits.data[i]));
            float td_error = q_continue - target_q_continue.data[i];

            // Gradient of sigmoid * td_error
            float sigmoid_grad = q_continue * (1.0f - q_continue);
            grad_continue.data[i] = 2.0f * td_error * sigmoid_grad;
        }
    }

    return {grad_halt, grad_continue};
}

// Helper: softmax
std::vector<float> LossFunctions::softmax(const std::vector<float>& logits) {
    std::vector<float> result = logits;
    float max_val = *std::max_element(result.begin(), result.end());

    // Subtract max for numerical stability
    for (auto& val : result) val -= max_val;

    // Exponentiate
    for (auto& val : result) val = std::exp(val);

    // Normalize
    float sum = std::accumulate(result.begin(), result.end(), 0.0f);
    for (auto& val : result) val /= sum;

    return result;
}

// Helper: log softmax
std::vector<float> LossFunctions::log_softmax(const std::vector<float>& logits) {
    std::vector<float> result = logits;
    float max_val = *std::max_element(result.begin(), result.end());

    // Subtract max for numerical stability
    for (auto& val : result) val -= max_val;

    // Log sum exp
    float sum_exp = 0.0f;
    for (auto val : result) sum_exp += std::exp(val);
    float log_sum_exp = std::log(sum_exp);

    // Log softmax
    for (auto& val : result) val -= log_sum_exp;

    return result;
}