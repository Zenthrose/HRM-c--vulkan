#pragma once

#include <memory>
#include <unordered_map>
#include <string>
#include <vector>
#include <cmath>
#include "../core/attention.hpp"  // For Tensor struct

/**
 * Character-Level Language Loss Functions
 *
 * Unlike token-based language models, this handles raw UTF-8 characters
 * directly without tokenization. This provides true multilingual support
 * and avoids tokenization artifacts.
 */
class CharacterLanguageLoss {
public:
    /**
     * Character-level cross-entropy loss
     * @param logits Model output logits (batch_size, seq_len, vocab_size)
     * @param targets Target character indices (batch_size, seq_len)
     * @param ignore_index Index to ignore in loss calculation (e.g., padding)
     * @return Loss tensor
     */
    static Tensor character_cross_entropy_loss(
        const Tensor& logits,
        const Tensor& targets,
        int ignore_index = -100
    );

    /**
     * Calculate character-level perplexity
     * @param loss Cross-entropy loss value
     * @param vocab_size Character vocabulary size (~100K for UTF-8)
     * @return Perplexity score
     */
    static float calculate_character_perplexity(float loss, int vocab_size = 100000);

    /**
     * Calculate character prediction accuracy
     * @param logits Model output logits
     * @param targets Target character indices
     * @param ignore_index Index to ignore
     * @return Accuracy percentage (0.0 to 1.0)
     */
    static float calculate_character_accuracy(
        const Tensor& logits,
        const Tensor& targets,
        int ignore_index = -100
    );

    /**
     * Calculate comprehensive language modeling metrics
     * @param logits Model output logits
     * @param targets Target character indices
     * @param vocab_size Character vocabulary size
     * @return Map of metric names to values
     */
    static std::unordered_map<std::string, float> calculate_metrics(
        const Tensor& logits,
        const Tensor& targets,
        int vocab_size = 100000,
        int ignore_index = -100
    );

    /**
     * Label smoothing for character predictions
     * @param logits Raw logits
     * @param targets Target indices
     * @param smoothing_factor Smoothing factor (0.0 to 1.0)
     * @param vocab_size Vocabulary size
     * @return Smoothed loss
     */
    static Tensor label_smoothed_character_loss(
        const Tensor& logits,
        const Tensor& targets,
        float smoothing_factor = 0.1f,
        int vocab_size = 100000
    );

private:
    // Helper functions for loss computation
    static Tensor softmax_cross_entropy_with_logits(const Tensor& logits, const Tensor& labels);
    static Tensor gather_logits_for_targets(const Tensor& logits, const Tensor& targets);
};