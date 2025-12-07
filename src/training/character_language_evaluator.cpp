#include "character_language_evaluator.hpp"
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <cmath>
#include <iostream>
#include <sstream>
#include <random>

CharacterLanguageEvaluator::CharacterLanguageEvaluator(std::shared_ptr<UTF8Processor> utf8_processor)
    : utf8_processor_(utf8_processor), loss_calculator_(std::make_unique<CharacterLanguageLoss>()),
      rng_(std::random_device{}()) {

    initialize_vocabulary();
    std::cout << "CharacterLanguageEvaluator initialized with vocabulary size: " << vocab_size_ << std::endl;
}

float CharacterLanguageEvaluator::evaluate_character_perplexity(const std::vector<std::string>& test_sequences) {
    if (test_sequences.empty()) {
        return 1.0f; // Perfect perplexity for empty input
    }

    float total_loss = 0.0f;
    size_t total_characters = 0;

    // For each test sequence, we would need model predictions
    // This is a simplified version that assumes we have model outputs
    // In practice, this would evaluate against actual model predictions

    for (const std::string& sequence : test_sequences) {
        if (sequence.length() < 2) continue; // Need at least 2 characters for prediction

        // Simplified perplexity calculation
        // In practice, this would use actual model logits
        float sequence_loss = std::log(static_cast<float>(vocab_size_)); // Uniform distribution baseline
        total_loss += sequence_loss * (sequence.length() - 1); // One prediction per character
        total_characters += (sequence.length() - 1);
    }

    if (total_characters == 0) return 1.0f;

    float avg_loss = total_loss / total_characters;
    return CharacterLanguageLoss::calculate_character_perplexity(avg_loss, vocab_size_);
}

std::string CharacterLanguageEvaluator::generate_text(const std::string& prompt, int max_length,
                                                    float temperature, float top_p) {
    std::string generated = prompt;

    // Convert prompt to character IDs
    std::vector<uint32_t> codepoints = utf8_processor_->encode_utf8(prompt);

    for (int i = 0; i < max_length && generated.length() < max_length; ++i) {
        // Get context window (last N characters)
        size_t context_start = (codepoints.size() > 100) ? codepoints.size() - 100 : 0;
        std::vector<char32_t> context(codepoints.begin() + context_start, codepoints.end());

        // In practice, this would call the model to get logits for next character
        // For now, we'll use a simple character frequency-based prediction
        std::vector<float> mock_logits(vocab_size_, 1.0f); // Uniform distribution

        // Apply temperature and top-p sampling
        int next_char_id = sample_next_character(mock_logits, temperature, top_p);

        if (next_char_id >= 0 && next_char_id < static_cast<int>(id_to_char_.size())) {
            char32_t next_char = id_to_char_[next_char_id];
            codepoints.push_back(next_char);

            // Convert back to UTF-8 and append
            std::vector<uint32_t> single_codepoint = {static_cast<uint32_t>(next_char)};
            std::string next_char_utf8 = utf8_processor_->decode_utf8(single_codepoint);
            generated += next_char_utf8;
        } else {
            break; // Invalid character ID
        }
    }

    return generated;
}

float CharacterLanguageEvaluator::calculate_character_accuracy(
    const std::vector<std::string>& predictions,
    const std::vector<std::string>& targets) {

    if (predictions.size() != targets.size()) {
        throw std::invalid_argument("Predictions and targets must have the same size");
    }

    size_t total_chars = 0;
    size_t correct_chars = 0;

    for (size_t i = 0; i < predictions.size(); ++i) {
        const std::string& pred = predictions[i];
        const std::string& target = targets[i];

        // Compare character by character
        size_t min_len = std::min(pred.length(), target.length());
        for (size_t j = 0; j < min_len; ++j) {
            if (pred[j] == target[j]) {
                correct_chars++;
            }
            total_chars++;
        }

        // Count extra characters in longer string as incorrect
        total_chars += std::abs(static_cast<int>(pred.length()) - static_cast<int>(target.length()));
    }

    return total_chars > 0 ? static_cast<float>(correct_chars) / total_chars : 0.0f;
}

std::unordered_map<std::string, float> CharacterLanguageEvaluator::evaluate_text_coherence(
    const std::string& generated_text) {

    std::unordered_map<std::string, float> metrics;

    if (generated_text.empty()) {
        metrics["coherence_score"] = 0.0f;
        return metrics;
    }

    // Character entropy (information content)
    metrics["character_entropy"] = calculate_character_entropy(generated_text);

    // N-gram diversity (avoids repetition)
    metrics["trigram_diversity"] = calculate_ngram_diversity(generated_text, 3);
    metrics["bigram_diversity"] = calculate_ngram_diversity(generated_text, 2);

    // Repetition score (lower is better)
    metrics["repetition_score"] = calculate_repetition_score(generated_text);

    // Overall coherence score (weighted combination)
    float coherence = 0.4f * metrics["character_entropy"] +
                     0.3f * metrics["trigram_diversity"] +
                     0.3f * (1.0f - metrics["repetition_score"]); // Invert repetition score

    metrics["coherence_score"] = std::max(0.0f, std::min(1.0f, coherence));

    return metrics;
}

std::unordered_map<std::string, float> CharacterLanguageEvaluator::calculate_language_metrics(
    const std::vector<std::string>& generated_texts,
    const std::vector<std::string>& reference_texts) {

    std::unordered_map<std::string, float> metrics;

    if (generated_texts.empty()) {
        return metrics;
    }

    // Basic statistics
    auto length_stats = calculate_length_stats(generated_texts);
    metrics.insert(length_stats.begin(), length_stats.end());

    // Character accuracy (if we have references)
    if (!reference_texts.empty() && reference_texts.size() == generated_texts.size()) {
        metrics["character_accuracy"] = calculate_character_accuracy(generated_texts, reference_texts);

        // BLEU-like score
        float total_bleu = 0.0f;
        for (size_t i = 0; i < generated_texts.size(); ++i) {
            total_bleu += calculate_character_bleu(generated_texts[i], reference_texts[i]);
        }
        metrics["character_bleu"] = total_bleu / generated_texts.size();
    }

    // Text coherence metrics
    float total_coherence = 0.0f;
    float total_entropy = 0.0f;
    float total_diversity = 0.0f;

    for (const std::string& text : generated_texts) {
        auto coherence_metrics = evaluate_text_coherence(text);
        total_coherence += coherence_metrics["coherence_score"];
        total_entropy += coherence_metrics["character_entropy"];
        total_diversity += coherence_metrics["trigram_diversity"];
    }

    size_t num_texts = generated_texts.size();
    metrics["avg_coherence"] = total_coherence / num_texts;
    metrics["avg_entropy"] = total_entropy / num_texts;
    metrics["avg_diversity"] = total_diversity / num_texts;

    return metrics;
}

std::unordered_map<std::string, float> CharacterLanguageEvaluator::evaluate_batch(
    const std::vector<Tensor>& model_logits,
    const std::vector<std::string>& target_sequences) {

    std::unordered_map<std::string, float> results;

    if (model_logits.size() != target_sequences.size()) {
        throw std::invalid_argument("Number of logits must match number of target sequences");
    }

    // Calculate loss metrics
    auto loss_metrics = CharacterLanguageLoss::calculate_metrics(
        model_logits[0], model_logits[0], vocab_size_); // Simplified for single batch
    results.insert(loss_metrics.begin(), loss_metrics.end());

    // Calculate text quality metrics
    auto quality_metrics = calculate_language_metrics(target_sequences);
    results.insert(quality_metrics.begin(), quality_metrics.end());

    return results;
}

// Private helper methods

void CharacterLanguageEvaluator::initialize_vocabulary() {
    // Initialize with basic ASCII characters
    for (char32_t c = 32; c < 127; ++c) {
        if (char_to_id_.find(c) == char_to_id_.end()) {
            char_to_id_[c] = id_to_char_.size();
            id_to_char_.push_back(c);
        }
    }

    // Add common Unicode characters
    std::vector<char32_t> common_unicode = {
        0x00A0, 0x00AD, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006,
        0x2007, 0x2008, 0x2009, 0x200A, 0x2010, 0x2011, 0x2012, 0x2013, 0x2014,
        0x2015, 0x2018, 0x2019, 0x201A, 0x201B, 0x201C, 0x201D, 0x201E, 0x201F,
        0x2026, 0x2030
    };

    for (char32_t c : common_unicode) {
        if (char_to_id_.find(c) == char_to_id_.end()) {
            char_to_id_[c] = id_to_char_.size();
            id_to_char_.push_back(c);
        }
    }

    vocab_size_ = id_to_char_.size();
}

int CharacterLanguageEvaluator::sample_next_character(const std::vector<float>& logits,
                                                    float temperature, float top_p) {
    if (logits.size() != static_cast<size_t>(vocab_size_)) {
        return -1; // Invalid logits size
    }

    // Apply temperature
    std::vector<float> tempered_logits = logits;
    if (temperature != 1.0f) {
        for (float& logit : tempered_logits) {
            logit /= temperature;
        }
    }

    // Apply nucleus sampling
    if (top_p < 1.0f) {
        tempered_logits = apply_nucleus_sampling(tempered_logits, top_p);
    }

    // Convert to probabilities
    float max_logit = *std::max_element(tempered_logits.begin(), tempered_logits.end());
    float sum_exp = 0.0f;
    for (float logit : tempered_logits) {
        sum_exp += std::exp(logit - max_logit);
    }

    std::vector<float> probabilities(vocab_size_);
    for (int i = 0; i < vocab_size_; ++i) {
        probabilities[i] = std::exp(tempered_logits[i] - max_logit) / sum_exp;
    }

    // Sample from distribution
    float random_value = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng_);
    float cumulative = 0.0f;

    for (int i = 0; i < vocab_size_; ++i) {
        cumulative += probabilities[i];
        if (random_value <= cumulative) {
            return i;
        }
    }

    return vocab_size_ - 1; // Fallback to last character
}

std::vector<float> CharacterLanguageEvaluator::apply_nucleus_sampling(
    const std::vector<float>& logits, float top_p) {

    // Sort indices by logit value (descending)
    std::vector<std::pair<float, int>> logit_indices;
    for (int i = 0; i < static_cast<int>(logits.size()); ++i) {
        logit_indices.emplace_back(logits[i], i);
    }
    std::sort(logit_indices.rbegin(), logit_indices.rend());

    // Find nucleus cutoff
    float cumulative_prob = 0.0f;
    float max_logit = logit_indices[0].first;
    int cutoff_idx = 0;

    for (int i = 0; i < static_cast<int>(logit_indices.size()); ++i) {
        float prob = std::exp(logit_indices[i].first - max_logit);
        cumulative_prob += prob;

        if (cumulative_prob >= top_p) {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Zero out logits below cutoff
    std::vector<float> filtered_logits = logits;
    for (int i = cutoff_idx; i < static_cast<int>(logit_indices.size()); ++i) {
        int original_idx = logit_indices[i].second;
        filtered_logits[original_idx] = -INFINITY;
    }

    return filtered_logits;
}

float CharacterLanguageEvaluator::calculate_character_entropy(const std::string& text) {
    if (text.empty()) return 0.0f;

    std::unordered_map<char, int> char_counts;
    for (char c : text) {
        char_counts[c]++;
    }

    float entropy = 0.0f;
    float text_len = static_cast<float>(text.length());

    for (const auto& pair : char_counts) {
        float prob = pair.second / text_len;
        entropy -= prob * std::log2(prob);
    }

    // Normalize by maximum possible entropy (log2 of unique characters)
    float max_entropy = std::log2(char_counts.size());
    return max_entropy > 0.0f ? entropy / max_entropy : 0.0f;
}

float CharacterLanguageEvaluator::calculate_ngram_diversity(const std::string& text, int n) {
    if (text.length() < n) return 0.0f;

    std::unordered_set<std::string> unique_ngrams;
    int total_ngrams = 0;

    for (size_t i = 0; i <= text.length() - n; ++i) {
        std::string ngram = text.substr(i, n);
        unique_ngrams.insert(ngram);
        total_ngrams++;
    }

    return total_ngrams > 0 ? static_cast<float>(unique_ngrams.size()) / total_ngrams : 0.0f;
}

float CharacterLanguageEvaluator::calculate_repetition_score(const std::string& text) {
    if (text.length() < 10) return 0.0f;

    // Count repeated character sequences
    int repeated_chars = 0;
    for (size_t i = 1; i < text.length(); ++i) {
        if (text[i] == text[i-1]) {
            repeated_chars++;
        }
    }

    return static_cast<float>(repeated_chars) / text.length();
}

std::unordered_map<std::string, float> CharacterLanguageEvaluator::calculate_length_stats(
    const std::vector<std::string>& texts) {

    std::unordered_map<std::string, float> stats;

    if (texts.empty()) {
        stats["avg_length"] = 0.0f;
        stats["min_length"] = 0.0f;
        stats["max_length"] = 0.0f;
        return stats;
    }

    std::vector<size_t> lengths;
    for (const std::string& text : texts) {
        lengths.push_back(text.length());
    }

    float sum = std::accumulate(lengths.begin(), lengths.end(), 0.0f);
    stats["avg_length"] = sum / lengths.size();
    stats["min_length"] = static_cast<float>(*std::min_element(lengths.begin(), lengths.end()));
    stats["max_length"] = static_cast<float>(*std::max_element(lengths.begin(), lengths.end()));

    return stats;
}

float CharacterLanguageEvaluator::calculate_character_bleu(const std::string& generated,
                                                         const std::string& reference) {
    // Simplified character-level BLEU-like score
    if (generated.empty() || reference.empty()) return 0.0f;

    // Character-level precision
    std::unordered_map<char, int> gen_counts, ref_counts, match_counts;

    for (char c : generated) gen_counts[c]++;
    for (char c : reference) ref_counts[c]++;

    for (char c : generated) {
        if (ref_counts[c] > 0) {
            match_counts[c] = std::min(gen_counts[c], ref_counts[c]);
        }
    }

    int total_matches = 0;
    int total_generated = generated.length();

    for (const auto& pair : match_counts) {
        total_matches += pair.second;
    }

    float precision = total_generated > 0 ? static_cast<float>(total_matches) / total_generated : 0.0f;

    // Length penalty (BLEU-style)
    float gen_len = generated.length();
    float ref_len = reference.length();
    float length_penalty = gen_len > ref_len ? 1.0f : std::exp(1.0f - ref_len / gen_len);

    return precision * length_penalty;
}

std::vector<float> CharacterLanguageEvaluator::logits_to_probabilities(const Tensor& logits_tensor) {
    // Convert logits tensor to probability distribution
    // This is a simplified implementation
    std::vector<float> probabilities;

    if (logits_tensor.shape.size() != 3 || logits_tensor.shape[2] != static_cast<uint32_t>(vocab_size_)) {
        return probabilities; // Invalid shape
    }

    // Take the last position's logits (next character prediction)
    size_t seq_len = logits_tensor.shape[1];
    size_t offset = (seq_len - 1) * vocab_size_;

    std::vector<float> logits(vocab_size_);
    for (int i = 0; i < vocab_size_; ++i) {
        logits[i] = logits_tensor.data[offset + i];
    }

    // Convert to probabilities
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum_exp = 0.0f;
    for (float logit : logits) {
        sum_exp += std::exp(logit - max_logit);
    }

    for (float logit : logits) {
        probabilities.push_back(std::exp(logit - max_logit) / sum_exp);
    }

    return probabilities;
}