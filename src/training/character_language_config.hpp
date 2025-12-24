#pragma once

#include <string>
#include <vector>
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
    float data_percentage = 1.0f;  // Percentage of dataset to use (0.0-1.0)

    // Training schedule
    int max_epochs = 100;
    int save_every_epochs = 5;
    int eval_every_steps = 1000;
    int warmup_steps = 1000;  // Linear warmup steps
    int total_steps = 100000;  // Total training steps
    float min_lr = 1e-6;  // Minimum learning rate

    // Parallel processing
    int parallel_batches = 1;  // Number of parallel batch processors (uses OpenMP if available)
    std::chrono::seconds max_training_time = std::chrono::hours(24);  // 24 hours max

};

// Progressive Data Feeding Configuration
struct DataStageConfig {
    int stage_id;
    std::string stage_name;
    float data_percentage;  // Percentage of total dataset (0.0-1.0)
    size_t max_sequences;   // Maximum sequences to load for this stage
    size_t memory_limit_mb; // Memory limit for this stage (MB)
    int context_length;     // Context window for this stage
    std::vector<std::string> allowed_file_types; // File types to include
    bool enable_compression; // Whether to use data compression
};

struct ProgressiveDataFeederConfig {
    std::vector<DataStageConfig> stages;
    size_t global_memory_limit_mb = 12288; // 12GB default
    size_t gpu_memory_limit_mb = 3072;     // 3GB default
    bool enable_prefetching = true;
    bool enable_compression = false;
    int prefetch_stages_ahead = 1;
    std::string data_root_path = "./data/text";
};