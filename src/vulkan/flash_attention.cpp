#include "flash_attention.hpp"
#include <random>
#include <thread>
#include <future>
#include <cstring>

FlashAttention::FlashAttention(const Config& config) : config_(config) {
    // Calculate block dimensions for tiling
    query_blocks_ = (config.seq_len + config.block_size - 1) / config.block_size;
    key_blocks_ = (config.seq_len + config.block_size - 1) / config.block_size;
    total_blocks_ = query_blocks_ * key_blocks_;

    std::cout << "FlashAttention initialized:" << std::endl;
    std::cout << "  Sequence length: " << config.seq_len << std::endl;
    std::cout << "  Block size: " << config.block_size << std::endl;
    std::cout << "  Query blocks: " << query_blocks_ << std::endl;
    std::cout << "  Key blocks: " << key_blocks_ << std::endl;
    std::cout << "  Theoretical speedup: " << get_theoretical_speedup() << "x" << std::endl;
}

std::vector<float> FlashAttention::forward(const std::vector<float>& query,
                                         const std::vector<float>& key,
                                         const std::vector<float>& value) {

    int batch_size = config_.batch_size;
    int num_heads = config_.num_heads;
    int seq_len = config_.seq_len;
    int head_dim = config_.head_dim;

    // Output tensor: [batch_size, num_heads, seq_len, head_dim]
    std::vector<float> output(batch_size * num_heads * seq_len * head_dim, 0.0f);

    // Process each batch and head independently
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            // Get pointers to current batch/head data
            const float* Q = query.data() + (b * num_heads * seq_len * head_dim) + (h * seq_len * head_dim);
            const float* K = key.data() + (b * num_heads * seq_len * head_dim) + (h * seq_len * head_dim);
            const float* V = value.data() + (b * num_heads * seq_len * head_dim) + (h * seq_len * head_dim);
            float* O = output.data() + (b * num_heads * seq_len * head_dim) + (h * seq_len * head_dim);

            // Temporary storage for FlashAttention algorithm
            std::vector<float> L(seq_len, 0.0f);  // Log-sum-exp normalization terms
            std::vector<float> M(seq_len, -std::numeric_limits<float>::infinity());  // Max values

            // Process query blocks
            for (int q_block = 0; q_block < query_blocks_; ++q_block) {
                int q_start = q_block * config_.block_size;
                int q_end = std::min(q_start + config_.block_size, seq_len);

                // Temporary output for this query block
                std::vector<float> O_block((q_end - q_start) * head_dim, 0.0f);
                std::vector<float> L_block(q_end - q_start, 0.0f);
                std::vector<float> M_block(q_end - q_start, -std::numeric_limits<float>::infinity());

                compute_query_block(Q + q_start * head_dim, q_start, q_end, K, V,
                                  O_block.data(), L_block.data(), M_block.data());

                // Update global output with this block's contribution
                for (int q = q_start; q < q_end; ++q) {
                    int local_q = q - q_start;
                    float scale = std::exp(M[q] - M_block[local_q]);

                    // Scale existing output
                    for (int d = 0; d < head_dim; ++d) {
                        O[q * head_dim + d] *= scale;
                    }

                    // Add new contribution
                    scale = std::exp(L[q] - L_block[local_q]);
                    for (int d = 0; d < head_dim; ++d) {
                        O[q * head_dim + d] += scale * O_block[local_q * head_dim + d];
                    }

                    // Update normalization terms
                    float new_m = std::max(M[q], M_block[local_q]);
                    float new_l = std::exp(M[q] - new_m) * L[q] + std::exp(M_block[local_q] - new_m) * L_block[local_q];

                    M[q] = new_m;
                    L[q] = new_l;

                    // Renormalize output
                    scale = std::exp(M[q] - new_m);
                    for (int d = 0; d < head_dim; ++d) {
                        O[q * head_dim + d] *= scale;
                    }
                }
            }

            // Final normalization
            for (int q = 0; q < seq_len; ++q) {
                float scale = std::exp(M[q] - L[q]);
                for (int d = 0; d < head_dim; ++d) {
                    O[q * head_dim + d] *= scale;
                }
            }
        }
    }

    return output;
}

void FlashAttention::compute_query_block(const float* Q_block, int q_start, int q_end,
                                       const float* K, const float* V,
                                       float* O_block, float* L_block, float* M_block) {

    int block_size = q_end - q_start;
    int seq_len = config_.seq_len;
    int head_dim = config_.head_dim;

    // Process key blocks
    for (int k_block = 0; k_block < key_blocks_; ++k_block) {
        int k_start = k_block * config_.block_size;
        int k_end = std::min(k_start + config_.block_size, seq_len);

        // Load K and V blocks into local memory (simulating fast memory)
        std::vector<float> K_local((k_end - k_start) * head_dim);
        std::vector<float> V_local((k_end - k_start) * head_dim);

        std::memcpy(K_local.data(), K + k_start * head_dim, K_local.size() * sizeof(float));
        std::memcpy(V_local.data(), V + k_end * head_dim, V_local.size() * sizeof(float));

        // Compute Q*K^T for this block
        std::vector<float> scores(block_size * (k_end - k_start), 0.0f);
        matmul_qk(Q_block, K_local.data(), scores.data(), block_size, k_end - k_start, head_dim);

        // Scale by 1/sqrt(head_dim)
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        for (auto& s : scores) s *= scale;

        // Apply causal masking if needed
        if (config_.causal) {
            apply_causal_mask(scores.data(), q_start, k_start, k_end, seq_len);
        }

        // Apply dropout if needed
        if (config_.dropout_prob > 0.0f) {
            apply_dropout(scores.data(), scores.size(), config_.dropout_prob);
        }

        // Incremental softmax computation
        for (int q = 0; q < block_size; ++q) {
            int global_q = q_start + q;
            std::vector<float> row_scores(k_end - k_start);
            for (int k = 0; k < k_end - k_start; ++k) {
                row_scores[k] = scores[q * (k_end - k_start) + k];
            }

            // Update running max and log-sum-exp
            float row_max = *std::max_element(row_scores.begin(), row_scores.end());
            float row_lse = 0.0f;
            for (float s : row_scores) {
                row_lse += std::exp(s - row_max);
            }
            row_lse = row_max + std::log(row_lse);

            // Update global statistics
            float new_m = std::max(M_block[q], row_lse);
            float scale_old = std::exp(M_block[q] - new_m);
            float scale_new = std::exp(row_lse - new_m);

            // Update output
            for (int d = 0; d < head_dim; ++d) {
                float pv_sum = 0.0f;
                for (int k = 0; k < k_end - k_start; ++k) {
                    float attn_prob = std::exp(row_scores[k] - row_max) / std::exp(row_lse - row_max);
                    pv_sum += attn_prob * V_local[k * head_dim + d];
                }
                O_block[q * head_dim + d] = O_block[q * head_dim + d] * scale_old + pv_sum * scale_new;
            }

            // Update normalization terms
            L_block[q] = std::log(scale_old * std::exp(L_block[q]) + scale_new);
            M_block[q] = new_m;
        }
    }
}

void FlashAttention::matmul_qk(const float* Q, const float* K, float* scores,
                              int q_rows, int k_cols, int head_dim) {
    // Q: [q_rows, head_dim], K: [k_cols, head_dim], scores: [q_rows, k_cols]
    for (int q = 0; q < q_rows; ++q) {
        for (int k = 0; k < k_cols; ++k) {
            float sum = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                sum += Q[q * head_dim + d] * K[k * head_dim + d];
            }
            scores[q * k_cols + k] = sum;
        }
    }
}

void FlashAttention::apply_causal_mask(float* scores, int q_idx, int k_start, int k_end, int seq_len) {
    int k_cols = k_end - k_start;
    for (int k = 0; k < k_cols; ++k) {
        int global_k = k_start + k;
        if (global_k > q_idx) {
            scores[k] = -std::numeric_limits<float>::infinity();
        }
    }
}

void FlashAttention::apply_dropout(float* scores, int size, float dropout_prob) {
    if (dropout_mask_.size() != static_cast<size_t>(size)) {
        generate_dropout_mask(size, dropout_prob);
    }

    for (int i = 0; i < size; ++i) {
        if (dropout_mask_[i]) {
            scores[i] = -std::numeric_limits<float>::infinity();
        }
    }
}

void FlashAttention::generate_dropout_mask(int size, float prob) {
    dropout_mask_.resize(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(prob);

    for (int i = 0; i < size; ++i) {
        dropout_mask_[i] = dist(gen);
    }
}

float FlashAttention::get_theoretical_speedup() const {
    // FlashAttention speedup is roughly seq_len / block_size
    // For typical values: seq_len=4096, block_size=256 -> 16x speedup
    return static_cast<float>(config_.seq_len) / config_.block_size;
}

// FlashAttentionV2 Implementation
FlashAttentionV2::FlashAttentionV2(const Config& config) : FlashAttention(config) {}

std::vector<float> FlashAttentionV2::forward_optimized(const std::vector<float>& query,
                                                     const std::vector<float>& key,
                                                     const std::vector<float>& value) {
    // Use parallel processing for better performance
    return forward(query, key, value);  // For now, delegate to base implementation
}

// LinearAttention Implementation
LinearAttention::LinearAttention(const Config& config) : config_(config) {}

std::vector<float> LinearAttention::forward(const std::vector<float>& query,
                                          const std::vector<float>& key,
                                          const std::vector<float>& value) {

    int batch_size = config_.batch_size;
    int num_heads = config_.num_heads;
    int seq_len = config_.seq_len;
    int head_dim = config_.head_dim;

    std::vector<float> output = value;  // Start with value as base

    // Apply feature map transformation if enabled
    if (config_.use_feature_map) {
        // Transform Q and K using ELU feature map
        std::vector<float> Q_transformed = query;
        std::vector<float> K_transformed = key;

        apply_feature_map(Q_transformed.data(), Q_transformed.size());
        apply_feature_map(K_transformed.data(), K_transformed.size());

        // Compute linear attention: (Q⊗K) * (K⊗V) / normalization
        linear_attention_kernel(Q_transformed.data(), K_transformed.data(),
                              value.data(), output.data());
    }

    return output;
}

void LinearAttention::apply_feature_map(float* x, int size) {
    // ELU feature map: x -> max(0, x) + min(0, exp(x) - 1)
    for (int i = 0; i < size; ++i) {
        if (x[i] > 0) {
            x[i] = x[i];  // max(0, x)
        } else {
            x[i] = std::exp(x[i]) - 1;  // exp(x) - 1 for x <= 0
        }
    }
}

void LinearAttention::linear_attention_kernel(const float* Q, const float* K, const float* V, float* O) {
    // Simplified linear attention computation
    // In practice, this would use more sophisticated kernels
    int seq_len = config_.seq_len;
    int head_dim = config_.head_dim;

    for (int q = 0; q < seq_len; ++q) {
        for (int d = 0; d < head_dim; ++d) {
            float sum = 0.0f;
            float norm = 0.0f;

            for (int k = 0; k < seq_len; ++k) {
                float qk_dot = Q[q * head_dim + d] * K[k * head_dim + d];
                sum += qk_dot * V[k * head_dim + d];
                norm += std::abs(qk_dot);
            }

            O[q * head_dim + d] = norm > 0 ? sum / norm : 0.0f;
        }
    }
}

// SparseAttention Implementation
SparseAttention::SparseAttention(const Config& config) : config_(config) {}

std::vector<float> SparseAttention::forward(const std::vector<float>& query,
                                          const std::vector<float>& key,
                                          const std::vector<float>& value) {

    // Create sparse attention mask
    auto sparse_mask = create_sparse_mask();

    // For now, return a copy of value (sparse computation would be implemented here)
    std::vector<float> output = value;

    // Apply sparse pattern (simplified)
    // In practice, this would modify attention computation to only use sparse connections

    return output;
}

std::vector<bool> SparseAttention::create_sparse_mask() {
    int seq_len = config_.seq_len;
    std::vector<bool> mask(seq_len * seq_len, false);

    if (config_.sparsity_pattern == "local") {
        // Local attention: attend to nearby tokens
        int block_size = config_.block_size;
        for (int i = 0; i < seq_len; ++i) {
            int start = std::max(0, i - block_size / 2);
            int end = std::min(seq_len, i + block_size / 2 + 1);
            for (int j = start; j < end; ++j) {
                mask[i * seq_len + j] = true;
            }
        }
    } else if (config_.sparsity_pattern == "strided") {
        // Strided attention: attend to every Nth token
        int stride = config_.block_size;
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; j += stride) {
                mask[i * seq_len + j] = true;
            }
        }
    }

    return mask;
}