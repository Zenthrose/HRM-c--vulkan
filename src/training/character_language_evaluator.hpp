#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <random>
#include <cmath>
#include "../core/attention.hpp"  // For Tensor struct
#include "../utils/utf8_processor.hpp"
#include "character_language_loss.hpp"

/**
 * Character-Level Language Model Evaluator
 *
 * Evaluates character-level language models with appropriate metrics
 * for raw UTF-8 text processing (not token-based evaluation).
 */
class CharacterLanguageEvaluator {
public:
    CharacterLanguageEvaluator(std::shared_ptr<UTF8Processor> utf8_processor);

    /**
     * Evaluate character-level perplexity on test sequences
     * @param test_sequences Vector of character sequences to evaluate
     * @return Perplexity score (lower is better)
     */
    float evaluate_character_perplexity(const std::vector<std::string>& test_sequences);

    /**
     * Generate text using character-by-character prediction
     * @param prompt Initial prompt text
     * @param max_length Maximum generation length in characters
     * @param temperature Sampling temperature (1.0 = neutral, <1.0 = conservative, >1.0 = creative)
     * @param top_p Nucleus sampling parameter (0.0-1.0, 1.0 = disabled)
     * @return Generated text
     */
    std::string generate_text(const std::string& prompt, int max_length = 500,
                            float temperature = 1.0f, float top_p = 0.9f);

    /**
     * Calculate character prediction accuracy
     * @param predictions Predicted character sequences
     * @param targets Target character sequences
     * @return Accuracy percentage (0.0 to 1.0)
     */
    float calculate_character_accuracy(const std::vector<std::string>& predictions,
                                     const std::vector<std::string>& targets);

    /**
     * Evaluate text coherence using character-level metrics
     * @param generated_text Text to evaluate
     * @return Map of coherence metrics
     */
    std::unordered_map<std::string, float> evaluate_text_coherence(
        const std::string& generated_text);

    /**
     * Calculate comprehensive language model quality metrics
     * @param generated_texts Generated text samples
     * @param reference_texts Reference text samples (for comparison)
     * @return Map of quality metrics
     */
    std::unordered_map<std::string, float> calculate_language_metrics(
        const std::vector<std::string>& generated_texts,
        const std::vector<std::string>& reference_texts = {});

    /**
     * Evaluate model on a batch of sequences
     * @param model_logits Model output logits for each sequence
     * @param target_sequences Target character sequences
     * @return Comprehensive evaluation results
     */
    std::unordered_map<std::string, float> evaluate_batch(
        const std::vector<Tensor>& model_logits,
        const std::vector<std::string>& target_sequences);

private:
    std::shared_ptr<UTF8Processor> utf8_processor_;
    std::unique_ptr<CharacterLanguageLoss> loss_calculator_;
    std::mt19937 rng_;

    // Character vocabulary (built from UTF-8 processor)
    std::unordered_map<char32_t, int> char_to_id_;
    std::vector<char32_t> id_to_char_;
    int vocab_size_;

    /**
     * Initialize character vocabulary from UTF-8 processor
     */
    void initialize_vocabulary();

    /**
     * Sample next character using temperature and top-p sampling
     * @param logits Logits for next character prediction
     * @param temperature Sampling temperature
     * @param top_p Nucleus sampling parameter
     * @return Selected character ID
     */
    int sample_next_character(const std::vector<float>& logits,
                            float temperature, float top_p);

    /**
     * Apply nucleus (top-p) sampling to logits
     * @param logits Input logits
     * @param top_p Probability threshold
     * @return Filtered and renormalized logits
     */
    std::vector<float> apply_nucleus_sampling(const std::vector<float>& logits, float top_p);

    /**
     * Calculate character entropy (information theoretic measure)
     * @param text Text to analyze
     * @return Character entropy value
     */
    float calculate_character_entropy(const std::string& text);

    /**
     * Calculate character n-gram diversity
     * @param text Text to analyze
     * @param n N-gram size
     * @return Diversity score (unique n-grams / total n-grams)
     */
    float calculate_ngram_diversity(const std::string& text, int n = 3);

    /**
     * Calculate character repetition score
     * @param text Text to analyze
     * @return Repetition score (lower is better)
     */
    float calculate_repetition_score(const std::string& text);

    /**
     * Calculate text length statistics
     * @param texts Collection of texts
     * @return Map of length statistics
     */
    std::unordered_map<std::string, float> calculate_length_stats(
        const std::vector<std::string>& texts);

    /**
     * Calculate BLEU-like score for character-level evaluation
     * @param generated Generated text
     * @param reference Reference text
     * @return BLEU score approximation
     */
    float calculate_character_bleu(const std::string& generated, const std::string& reference);

    /**
     * Convert logits tensor to probability distribution
     * @param logits_tensor Model output logits
     * @return Probability distribution vector
     */
    std::vector<float> logits_to_probabilities(const Tensor& logits_tensor);
};