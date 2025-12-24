#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include "../vulkan/flash_attention.hpp"
#include "../core/attention.hpp"  // For Tensor struct

/**
 * Mixed Precision Training Configuration
 */
struct MixedPrecisionConfig {
    bool enabled = true;
    bool use_fp16 = true;
    bool use_bf16 = false;  // Better for some GPUs
    float loss_scale = 65536.0f;
    bool dynamic_loss_scaling = true;
    int gradient_clip_norm = 1;
};

/**
 * Gradient Checkpointing Configuration
 */
struct GradientCheckpointingConfig {
    bool enabled = true;
    int checkpoint_interval = 100;  // Checkpoint every N layers
    bool preserve_rng_state = true;
    size_t memory_budget_mb = 8192;  // 8GB memory budget
};

/**
 * Advanced Optimizer Configuration
 */
struct AdvancedOptimizerConfig {
    std::string optimizer_type = "adamw";  // adamw, lion, adafactor
    float learning_rate = 1e-4f;
    float weight_decay = 0.01f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;

    // Lion optimizer specific
    float lion_beta1 = 0.95f;
    float lion_beta2 = 0.98f;

    // Adafactor specific
    bool adafactor_relative_step = true;
    float adafactor_scale_parameter = 1.0f;
    float adafactor_warmup_init = 0.0f;
};

/**
 * Learning Rate Scheduler Configuration
 */
struct LRSchedulerConfig {
    std::string scheduler_type = "cosine";  // cosine, linear, warmup_cosine
    size_t warmup_steps = 2000;
    size_t total_steps = 100000;
    float max_lr = 1e-3f;
    float min_lr = 1e-6f;
    float decay_rate = 0.1f;
    size_t decay_steps = 30000;
};

/**
 * Advanced Training Optimizations for HRM
 *
 * Implements cutting-edge training techniques for improved performance
 * and efficiency in large-scale language model training.
 *
 * Key Features:
 * - FlashAttention: O(n) attention computation
 * - Mixed Precision: FP16/FP8 training support
 * - Gradient Checkpointing: Memory-efficient training
 * - Advanced Optimizers: Lion, Adafactor, etc.
 */
class AdvancedTrainingOptimizations {
public:

    // Core optimization methods
    static void enable_mixed_precision_training(const MixedPrecisionConfig& config);
    static void enable_gradient_checkpointing(const GradientCheckpointingConfig& config);
    static void setup_advanced_optimizer(const AdvancedOptimizerConfig& config);
    static void setup_lr_scheduler(const LRSchedulerConfig& config);

    // Memory optimization
    static size_t estimate_memory_usage_with_optimizations(
        size_t model_params, size_t batch_size, size_t seq_len,
        const MixedPrecisionConfig& mp_config = {},
        const GradientCheckpointingConfig& gc_config = {});

    // Performance monitoring
    static std::unordered_map<std::string, float> get_training_metrics();
    static void log_optimization_stats();

private:
    // Internal state - defined in .cpp file
    static MixedPrecisionConfig current_mp_config_;
    static GradientCheckpointingConfig current_gc_config_;
    static AdvancedOptimizerConfig current_opt_config_;
    static LRSchedulerConfig current_lr_config_;

    // Helper methods
    static float compute_adaptive_loss_scale(float current_loss, float prev_loss);
    static std::vector<size_t> compute_checkpoint_boundaries(
        int num_layers, int checkpoint_interval, size_t memory_budget);
    static float compute_learning_rate(size_t step, const LRSchedulerConfig& config);
};

/**
 * FlashAttention Configuration
 */
struct FlashAttentionConfig {
    bool enabled = true;
    int block_size = 256;  // Block size for tiling
    bool causal = true;    // Causal attention for language
    bool use_fp16 = true;  // Use half precision
    int num_splits = 1;    // Number of splits for memory efficiency
};

/**
 * Sparse Attention Configuration
 */
struct SparseAttentionConfig {
    bool enabled = true;
    float sparsity_ratio = 0.1f;  // Keep 10% of attention weights
    std::string sparsity_pattern = "random";  // random, local, global
    bool dynamic_sparsity = true;  // Adapt sparsity during training
    int block_size = 32;
};

/**
 * Linear Attention Configuration (O(n) complexity)
 */
struct LinearAttentionConfig {
    bool enabled = true;
    int feature_map_dim = 256;  // Dimension for feature map
    std::string feature_map_type = "elu";  // elu, relu, identity
    bool use_kernel_trick = true;
};

/**
 * Memory-Efficient Attention Mechanisms
 *
 * Implements FlashAttention, sparse attention, and other
 * efficient attention variants for 2-10x speedup.
 */
class EfficientAttention {
public:

    // Efficient attention implementations
    static Tensor flash_attention_forward(
        const Tensor& query, const Tensor& key, const Tensor& value,
        const FlashAttentionConfig& config);

    static Tensor sparse_attention_forward(
        const Tensor& query, const Tensor& key, const Tensor& value,
        const SparseAttentionConfig& config);

    static Tensor linear_attention_forward(
        const Tensor& query, const Tensor& key, const Tensor& value,
        const LinearAttentionConfig& config);

    // Memory-efficient RoPE
    static void apply_rotary_embeddings_efficient(
        Tensor& tensor, int position, bool use_half_precision = true);

    // Performance utilities
    static std::unordered_map<std::string, float> benchmark_attention_methods(
        int batch_size, int seq_len, int num_heads, int head_dim);

    static size_t estimate_attention_memory_usage(
        int batch_size, int seq_len, int num_heads, int head_dim,
        const std::string& attention_type = "flash");

private:
    // CUDA kernel interfaces (would be implemented in .cu files)
    static void flash_attention_cuda_kernel(
        const float* query, const float* key, const float* value,
        float* output, int batch_size, int seq_len, int num_heads, int head_dim,
        bool causal, void* stream);

    static void sparse_attention_cuda_kernel(
        const float* query, const float* key, const float* value,
        float* output, const int* sparsity_mask,
        int batch_size, int seq_len, int num_heads, int head_dim,
        void* stream);
};

/**
 * Multi-GPU Configuration
 */
struct MultiGPUConfig {
    int num_gpus = 1;
    bool use_nccl = true;  // NVIDIA Collective Communications Library
    int gradient_accumulation_steps = 1;
    bool zero_optimization = false;  // ZeRO optimization
    int zero_stage = 1;  // ZeRO stage (1, 2, or 3)
};

/**
 * Model Parallelism Configuration
 */
struct ModelParallelConfig {
    int tensor_parallel_size = 1;  // Split tensors across GPUs
    int pipeline_parallel_size = 1;  // Split layers across GPUs
    bool use_megatron_style = true;  // Megatron-LM style parallelism
    int num_micro_batches = 1;
};

/**
 * Data Parallelism Configuration
 */
struct DataParallelConfig {
    bool enabled = true;
    bool use_ddp = true;  // DistributedDataParallel
    bool gradient_as_bucket_view = true;  // Memory optimization
    int bucket_size_mb = 25;  // Gradient bucket size
};

/**
 * Distributed Training Infrastructure
 *
 * Enables training on multiple GPUs and machines for
 * models 100x larger than single-GPU training.
 */
class DistributedTrainingManager {
public:

    // Initialization
    static void initialize_multi_gpu_training(const MultiGPUConfig& config);
    static void setup_model_parallelism(const ModelParallelConfig& config);
    static void setup_data_parallelism(const DataParallelConfig& config);

    // Gradient synchronization
    static void all_reduce_gradients(const std::vector<Tensor>& gradients);
    static void all_reduce_gradients_async(const std::vector<Tensor>& gradients);

    // Communication optimization
    static void optimize_communication_overhead();
    static void setup_gradient_compression(float compression_ratio = 0.01f);

    // Monitoring and debugging
    static std::unordered_map<std::string, float> get_distributed_metrics();
    static void synchronize_all_gpus();
    static bool check_gradient_health(const std::vector<Tensor>& gradients);

    // Fault tolerance
    static void setup_fault_tolerance();
    static void handle_gpu_failure(int failed_gpu_id);
    static void redistribute_workload();

private:
    static MultiGPUConfig current_gpu_config_;
    static ModelParallelConfig current_mp_config_;
    static DataParallelConfig current_dp_config_;

    // NCCL communication handles
    static void* nccl_comm_;
    static void* nccl_stream_;

    // Helper methods
    static void initialize_nccl(int num_gpus);
    static void setup_zero_optimization(int stage);
    static void create_model_parallel_groups(int tensor_parallel_size, int pipeline_parallel_size);
};