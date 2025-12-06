#pragma once

#include <string>
#include <chrono>

/**
 * Configuration for Character-Level Language Model Training
 *
 * Unlike token-based models, this handles raw UTF-8 characters directly,
 * providing true multilingual support without tokenization artifacts.
 */
struct CharacterLanguageModelConfig {
    // Model architecture (optimized for character-level processing)
    int char_vocab_size = 100000;  // UTF-8 character vocabulary (much larger than tokens)
    int hidden_size = 768;
    int num_layers = 4;  // Original HRM design: 4 layers per reasoning module
    int num_heads = 12;
    int max_seq_length = 2048;  // Longer sequences for character-level models

    // Character processing
    bool use_utf8_normalization = true;
    std::string text_encoding = "utf-8";
    int min_char_frequency = 1;  // Minimum frequency for character to be included in vocab

    // Training parameters (adjusted for character-level)
    float learning_rate = 5e-5;  // Lower learning rate for character models
    int batch_size = 4;  // Smaller batches due to longer sequences
    int gradient_accumulation_steps = 8;  // More accumulation for effective batch size
    float weight_decay = 0.01f;
    float max_grad_norm = 1.0f;

    // Loss and optimization
    std::string loss_type = "character_cross_entropy";
    std::string optimizer = "adamw";
    float label_smoothing = 0.1f;
    bool use_mixed_precision = false;

    // Data processing
    std::string dataset_path = "./data/text";
    float train_val_split = 0.9f;
    int context_length = 1024;  // Context window for next-character prediction
    bool shuffle_sequences = true;

    // Training schedule
    int max_epochs = 100;
    int save_every_epochs = 5;
    int eval_every_steps = 1000;
    int warmup_steps = 1000;  // Linear warmup steps
    int total_steps = 100000;  // Total training steps
    float min_lr = 1e-6;  // Minimum learning rate
    std::chrono::seconds max_training_time = std::chrono::hours(24);  // 24 hours max

    // Early stopping
    int patience = 10;
    float min_improvement = 1e-4;
    std::string early_stopping_metric = "val_perplexity";

    // Generation parameters
    float generation_temperature = 1.0f;
    float top_p = 0.9f;
    int max_generation_length = 500;
    bool use_nucleus_sampling = true;

    // Memory optimization
    bool use_gradient_checkpointing = false;
    bool offload_activations = false;
    size_t max_memory_usage_mb = 8192;  // 8GB limit

    // Logging and monitoring
    std::string log_dir = "./logs";
    std::string checkpoint_dir = "./checkpoints";
    bool log_gradients = false;
    bool profile_memory = true;

    // Distributed training (for future expansion)
    bool use_distributed_training = false;
    int world_size = 1;
    int local_rank = 0;

    /**
     * Validate configuration parameters
     * @return true if configuration is valid
     */
    bool validate() const {
        if (char_vocab_size <= 0 || char_vocab_size > 1000000) {
            return false;
        }
        if (hidden_size <= 0 || hidden_size > 10000) {
            return false;
        }
        if (num_layers <= 0 || num_layers > 100) {
            return false;
        }
        if (num_heads <= 0 || num_heads > 100) {
            return false;
        }
        if (max_seq_length <= 0 || max_seq_length > 10000) {
            return false;
        }
        if (batch_size <= 0 || batch_size > 1000) {
            return false;
        }
        if (learning_rate <= 0.0f || learning_rate > 1.0f) {
            return false;
        }
        if (context_length <= 0 || context_length > max_seq_length) {
            return false;
        }
        if (train_val_split <= 0.0f || train_val_split >= 1.0f) {
            return false;
        }
        return true;
    }

    /**
     * Get human-readable description of configuration
     * @return Configuration summary string
     */
    std::string get_description() const {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer),
            "Character Language Model Config:\n"
            "  Vocab Size: %d characters\n"
            "  Hidden Size: %d\n"
            "  Layers: %d, Heads: %d\n"
            "  Max Seq Length: %d\n"
            "  Context Length: %d\n"
            "  Batch Size: %d (effective: %d)\n"
            "  Learning Rate: %.2e\n"
            "  Label Smoothing: %.2f",
            char_vocab_size, hidden_size, num_layers, num_heads,
            max_seq_length, context_length, batch_size,
            batch_size * gradient_accumulation_steps,
            learning_rate, label_smoothing);
        return std::string(buffer);
    }

    /**
     * Calculate effective batch size (accounting for gradient accumulation)
     * @return Effective batch size for training
     */
    int get_effective_batch_size() const {
        return batch_size * gradient_accumulation_steps;
    }

    /**
     * Estimate memory requirements for this configuration
     * @return Estimated memory usage in MB
     */
    size_t estimate_memory_usage_mb() const {
        // Rough estimation: model parameters + activations + gradients
        size_t param_memory = (char_vocab_size * hidden_size +  // embeddings
                              num_layers * 12 * hidden_size * hidden_size +  // attention + MLP
                              num_layers * hidden_size) * 4;  // float32

        size_t activation_memory = batch_size * max_seq_length * hidden_size * num_layers * 4;
        size_t gradient_memory = param_memory;  // Same size as parameters

        return (param_memory + activation_memory + gradient_memory) / (1024 * 1024);
    }
};