#include "character_language_loss.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <filesystem>

Tensor CharacterLanguageLoss::character_cross_entropy_loss(
    const Tensor& logits,
    const Tensor& targets,
    int ignore_index
) {
    // Character-level cross-entropy loss
    // logits: (batch_size, seq_len, vocab_size)
    // targets: (batch_size, seq_len)

    if (logits.shape.size() != 3 || targets.shape.size() != 2) {
        throw std::invalid_argument("Invalid tensor shapes for character cross-entropy loss");
    }

    size_t batch_size = logits.shape[0];
    size_t seq_len = logits.shape[1];
    size_t vocab_size = logits.shape[2];

    if (targets.shape[0] != batch_size || targets.shape[1] != seq_len) {
        throw std::invalid_argument("Logits and targets shape mismatch");
    }

    Tensor loss;
    loss.shape = {static_cast<uint32_t>(batch_size), static_cast<uint32_t>(seq_len)};
    loss.data.resize(batch_size * seq_len, 0.0f);

    float total_loss = 0.0f;
    size_t valid_count = 0;

    // Compute cross-entropy for each position
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            size_t target_idx = static_cast<size_t>(targets.data[b * seq_len + s]);

            // Skip ignored indices (padding, etc.)
            if (target_idx == static_cast<size_t>(ignore_index)) {
                loss.data[b * seq_len + s] = 0.0f;
                continue;
            }

            if (target_idx >= vocab_size) {
                throw std::invalid_argument("Target index out of vocabulary range");
            }

            // Get logits for this position
            size_t logits_offset = b * seq_len * vocab_size + s * vocab_size;

            // Find max logit for numerical stability
            float max_logit = *std::max_element(
                logits.data.begin() + logits_offset,
                logits.data.begin() + logits_offset + vocab_size
            );

            // Compute softmax and cross-entropy (with bounds checking)
            float sum_exp = 0.0f;
            for (size_t v = 0; v < vocab_size; ++v) {
                size_t idx = logits_offset + v;
                if (idx >= logits.data.size()) {
                    std::cerr << "Error: logits index out of bounds: " << idx << " >= " << logits.data.size() << std::endl;
                    continue;
                }
                sum_exp += std::exp(logits.data[idx] - max_logit);
            }

            float log_prob = logits.data[logits_offset + target_idx] - max_logit - std::log(sum_exp);
            float position_loss = -log_prob;

            loss.data[b * seq_len + s] = position_loss;
            total_loss += position_loss;
            valid_count++;
        }
    }

    // Return average loss
    if (valid_count > 0) {
        for (auto& val : loss.data) {
            val /= valid_count;
        }
    }

    return loss;
}

float CharacterLanguageLoss::calculate_character_perplexity(float loss, int vocab_size) {
    // Character-level perplexity: exp(cross_entropy_loss)
    // Note: This is different from token-level perplexity
    return std::exp(loss);
}

float CharacterLanguageLoss::calculate_character_accuracy(
    const Tensor& logits,
    const Tensor& targets,
    int ignore_index
) {
    if (logits.shape.size() != 3 || targets.shape.size() != 2) {
        throw std::invalid_argument("Invalid tensor shapes for character accuracy");
    }

    size_t batch_size = logits.shape[0];
    size_t seq_len = logits.shape[1];
    size_t vocab_size = logits.shape[2];

    size_t correct_predictions = 0;
    size_t total_predictions = 0;

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            size_t target_idx = static_cast<size_t>(targets.data[b * seq_len + s]);

            if (target_idx == static_cast<size_t>(ignore_index)) {
                continue;
            }

            // Find predicted character (argmax)
            size_t logits_offset = b * seq_len * vocab_size + s * vocab_size;
            size_t predicted_idx = 0;
            float max_logit = logits.data[logits_offset];

            for (size_t v = 1; v < vocab_size; ++v) {
                if (logits.data[logits_offset + v] > max_logit) {
                    max_logit = logits.data[logits_offset + v];
                    predicted_idx = v;
                }
            }

            if (predicted_idx == target_idx) {
                correct_predictions++;
            }
            total_predictions++;
        }
    }

    return total_predictions > 0 ? static_cast<float>(correct_predictions) / total_predictions : 0.0f;
}

std::unordered_map<std::string, float> CharacterLanguageLoss::calculate_metrics(
    const Tensor& logits,
    const Tensor& targets,
    int vocab_size,
    int ignore_index
) {
    std::unordered_map<std::string, float> metrics;

    // Calculate cross-entropy loss
    Tensor loss_tensor = character_cross_entropy_loss(logits, targets, ignore_index);

    // Average loss across all positions
    float total_loss = 0.0f;
    size_t valid_count = 0;
    for (float val : loss_tensor.data) {
        if (val > 0.0f) {  // Only count non-ignored positions
            total_loss += val;
            valid_count++;
        }
    }
    float avg_loss = valid_count > 0 ? total_loss / valid_count : 0.0f;

    // Calculate perplexity
    float perplexity = calculate_character_perplexity(avg_loss, vocab_size);

    // Calculate accuracy
    float accuracy = calculate_character_accuracy(logits, targets, ignore_index);

    metrics["character_cross_entropy_loss"] = avg_loss;
    metrics["character_perplexity"] = perplexity;
    metrics["character_accuracy"] = accuracy;

    return metrics;
}

Tensor CharacterLanguageLoss::label_smoothed_character_loss(
    const Tensor& logits,
    const Tensor& targets,
    float smoothing_factor,
    int vocab_size
) {
    // Label smoothing for character predictions
    // smoothing_factor: amount of probability mass to distribute (0.0 to 1.0)

    if (logits.shape.size() != 3 || targets.shape.size() != 2) {
        throw std::invalid_argument("Invalid tensor shapes for label smoothing");
    }

    size_t batch_size = logits.shape[0];
    size_t seq_len = logits.shape[1];

    Tensor smoothed_targets;
    smoothed_targets.shape = logits.shape;  // Same shape as logits
    smoothed_targets.data.resize(logits.data.size(), 0.0f);

    // Create smoothed target distribution
    float uniform_prob = smoothing_factor / vocab_size;
    float correct_prob = 1.0f - smoothing_factor + uniform_prob;

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            size_t target_idx = static_cast<size_t>(targets.data[b * seq_len + s]);
            size_t targets_offset = b * seq_len * vocab_size + s * vocab_size;

            // Set uniform probability for all characters
            for (size_t v = 0; v < static_cast<size_t>(vocab_size); ++v) {
                smoothed_targets.data[targets_offset + v] = uniform_prob;
            }

            // Add extra probability mass to correct character
            if (target_idx < static_cast<size_t>(vocab_size)) {
                smoothed_targets.data[targets_offset + target_idx] = correct_prob;
            }
        }
    }

    // Compute KL divergence between predicted and smoothed targets
    return softmax_cross_entropy_with_logits(logits, smoothed_targets);
}

Tensor CharacterLanguageLoss::softmax_cross_entropy_with_logits(
    const Tensor& logits,
    const Tensor& labels
) {
    // Compute softmax cross-entropy: -sum(labels * log(softmax(logits)))

    if (logits.shape != labels.shape) {
        throw std::invalid_argument("Logits and labels must have the same shape");
    }

    Tensor loss;
    loss.shape = {logits.shape[0], logits.shape[1]};  // (batch_size, seq_len)
    loss.data.resize(logits.shape[0] * logits.shape[1], 0.0f);

    size_t batch_size = logits.shape[0];
    size_t seq_len = logits.shape[1];
    size_t vocab_size = logits.shape[2];

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            size_t offset = b * seq_len * vocab_size + s * vocab_size;

            // Find max logit for numerical stability
            float max_logit = *std::max_element(
                logits.data.begin() + offset,
                logits.data.begin() + offset + vocab_size
            );

            // Compute softmax
            std::vector<float> softmax_probs(vocab_size);
            float sum_exp = 0.0f;

            for (size_t v = 0; v < vocab_size; ++v) {
                float exp_val = std::exp(logits.data[offset + v] - max_logit);
                softmax_probs[v] = exp_val;
                sum_exp += exp_val;
            }

            for (size_t v = 0; v < vocab_size; ++v) {
                softmax_probs[v] /= sum_exp;
            }

            // Compute cross-entropy
            float position_loss = 0.0f;
            for (size_t v = 0; v < vocab_size; ++v) {
                float target_prob = labels.data[offset + v];
                if (target_prob > 0.0f) {
                    position_loss -= target_prob * std::log(softmax_probs[v] + 1e-10f);
                }
            }

            loss.data[b * seq_len + s] = position_loss;
        }
    }

    return loss;
}