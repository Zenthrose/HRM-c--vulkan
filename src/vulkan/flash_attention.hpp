#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>

/**
 * FlashAttention Implementation
 *
 * Implements the FlashAttention algorithm for efficient attention computation.
 * Key innovations:
 * - O(n) time complexity instead of O(n²)
 * - O(n) memory usage instead of O(n²)
 * - Tiling strategy to fit in fast memory
 * - Incremental softmax computation
 */
class FlashAttention {
public:
    struct Config {
        int batch_size;
        int num_heads;
        int seq_len;
        int head_dim;
        int block_size;  // Tile size for FlashAttention (typically 256-512)
        bool causal;     // Whether to apply causal masking
        float dropout_prob;  // Dropout probability (0.0 = no dropout)
        bool use_softmax;    // Whether to apply softmax normalization
    };

    FlashAttention(const Config& config);
    ~FlashAttention() = default;

    /**
     * Forward pass of FlashAttention
     * @param query Query tensor [batch_size, num_heads, seq_len, head_dim]
     * @param key Key tensor [batch_size, num_heads, seq_len, head_dim]
     * @param value Value tensor [batch_size, num_heads, seq_len, head_dim]
     * @return Output tensor [batch_size, num_heads, seq_len, head_dim]
     */
    std::vector<float> forward(const std::vector<float>& query,
                              const std::vector<float>& key,
                              const std::vector<float>& value);

    /**
     * Get configuration
     */
    const Config& get_config() const { return config_; }

    /**
     * Get theoretical speedup factor for current configuration
     */
    float get_theoretical_speedup() const;

private:
    Config config_;

    // Pre-computed constants for efficiency
    int query_blocks_;  // Number of query blocks
    int key_blocks_;    // Number of key blocks
    int total_blocks_;  // Total number of blocks to process

    /**
     * Compute attention for a single query block
     * This is the core of FlashAttention algorithm
     */
    void compute_query_block(const float* Q_block, int q_start, int q_end,
                           const float* K, const float* V,
                           float* O_block, float* L_block, float* M_block);

    /**
     * Compute softmax incrementally for a block
     */
    void incremental_softmax(float* scores, int size, float* L, float* M);

    /**
     * Apply causal masking to attention scores
     */
    void apply_causal_mask(float* scores, int q_idx, int k_start, int k_end, int seq_len);

    /**
     * Matrix multiplication helper (Q*K^T)
     */
    void matmul_qk(const float* Q, const float* K, float* scores,
                  int q_rows, int k_cols, int head_dim);

    /**
     * Matrix multiplication helper (P*V)
     */
    void matmul_pv(const float* P, const float* V, float* O,
                  int rows, int cols, int head_dim);

    /**
     * Scale and add operation: O = O * scale + new_O
     */
    void scale_add(float* O, const float* new_O, float scale, int size);

    /**
     * Apply dropout to attention scores
     */
    void apply_dropout(float* scores, int size, float dropout_prob);

    // Random number generation for dropout
    std::vector<float> dropout_mask_;
    void generate_dropout_mask(int size, float prob);
};

/**
 * FlashAttentionV2 - Enhanced version with further optimizations
 */
class FlashAttentionV2 : public FlashAttention {
public:
    FlashAttentionV2(const Config& config);

    /**
     * Forward pass with additional optimizations
     */
    std::vector<float> forward_optimized(const std::vector<float>& query,
                                       const std::vector<float>& key,
                                       const std::vector<float>& value);

private:
    /**
     * Parallel block processing for better cache utilization
     */
    void process_blocks_parallel(const float* Q, const float* K, const float* V,
                               float* O, int num_threads = 4);

    /**
     * Memory-efficient block processing with streaming
     */
    void streaming_attention(const float* Q, const float* K, const float* V, float* O);
};

/**
 * Linear Attention - O(n) complexity alternative
 */
class LinearAttention {
public:
    struct Config {
        int batch_size;
        int num_heads;
        int seq_len;
        int head_dim;
        std::string kernel_type;  // "elu", "relu", "softplus"
        bool use_feature_map;     // Whether to use feature map transformation
    };

    LinearAttention(const Config& config);

    std::vector<float> forward(const std::vector<float>& query,
                              const std::vector<float>& key,
                              const std::vector<float>& value);

private:
    Config config_;

    /**
     * Apply feature map transformation (ELU kernel)
     */
    void apply_feature_map(float* x, int size);

    /**
     * Compute linear attention: (Q⊗K) * (K⊗V) / normalization
     */
    void linear_attention_kernel(const float* Q, const float* K, const float* V, float* O);
};

/**
 * Sparse Attention - Dynamic sparsity for long contexts
 */
class SparseAttention {
public:
    struct Config {
        int batch_size;
        int num_heads;
        int seq_len;
        int head_dim;
        int block_size;           // Local attention block size
        int global_tokens;        // Number of global tokens to attend to
        std::string sparsity_pattern;  // "local", "strided", "random", "learned"
    };

    SparseAttention(const Config& config);

    std::vector<float> forward(const std::vector<float>& query,
                              const std::vector<float>& key,
                              const std::vector<float>& value);

private:
    Config config_;

    /**
     * Create sparse attention mask based on pattern
     */
    std::vector<bool> create_sparse_mask();

    /**
     * Apply sparse attention pattern
     */
    void apply_sparse_pattern(float* attention_scores, const std::vector<bool>& mask);
};