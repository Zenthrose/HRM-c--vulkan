#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <string>

namespace Nyx {

// Forward declarations
class INT4Model;
struct TrainingBatch;
struct QuantizationConfig;

// Gradient checkpoint for memory-efficient training
struct GradientCheckpoint {
    std::string checkpoint_id;
    int layer_index;
    std::vector<float> input_activations;  // Store inputs instead of outputs
    std::unordered_map<std::string, std::vector<float>> intermediate_states;
    size_t memory_usage;
    bool is_valid;
};

// Memory-efficient training with gradient checkpointing
class GradientCheckpointingManager {
public:
    GradientCheckpointingManager(size_t max_memory_mb = 512);

    // Checkpoint management
    void create_checkpoint(int layer_index, const std::vector<float>& input,
                          const std::string& checkpoint_id);
    void save_checkpoint(const GradientCheckpoint& checkpoint);
    GradientCheckpoint load_checkpoint(const std::string& checkpoint_id);

    // Recomputation for backward pass
    std::vector<float> recompute_forward(const GradientCheckpoint& checkpoint,
                                       const INT4Model& model);
    std::vector<float> compute_checkpoint_gradient(const GradientCheckpoint& checkpoint,
                                                 const std::vector<float>& output_gradient,
                                                 const INT4Model& model);

    // Memory optimization
    void optimize_checkpoint_schedule(const INT4Model& model,
                                   const TrainingBatch& batch);
    size_t estimate_memory_savings(const INT4Model& model) const;

    // Checkpoint cleanup
    void clear_checkpoints();
    void remove_old_checkpoints(size_t keep_recent = 5);

private:
    std::unordered_map<std::string, GradientCheckpoint> checkpoints_;
    size_t max_memory_mb_;
    size_t current_memory_usage_;

    // Checkpoint scheduling
    std::vector<int> optimal_checkpoint_indices_;
    void calculate_optimal_schedule(const INT4Model& model);

    // Memory estimation
    size_t estimate_checkpoint_memory(const GradientCheckpoint& checkpoint) const;
    size_t estimate_recomputation_cost(const INT4Model& model, int start_layer, int end_layer) const;
};

// Activation offloading for CPU RAM
class ActivationOffloadingManager {
public:
    ActivationOffloadingManager(size_t max_cpu_memory_mb = 2048);

    // Offload activations to CPU RAM
    void offload_activations(const std::vector<std::vector<float>>& activations,
                           const std::vector<int>& layer_indices);
    std::vector<float> reload_activations(int layer_index);

    // Intelligent offloading decisions
    bool should_offload_layer(int layer_index, size_t activation_size,
                            const std::vector<float>& current_usage) const;

    // Memory management
    void optimize_offloading_strategy(const INT4Model& model);
    size_t get_current_offload_memory() const;

    // Prefetching for efficiency
    void prefetch_activations(const std::vector<int>& upcoming_layers);
    void evict_unused_activations();

private:
    std::unordered_map<int, std::vector<float>> offloaded_activations_;
    size_t max_cpu_memory_mb_;
    size_t current_usage_mb_;

    // Offloading policies
    std::vector<int> layers_to_offload_;
    void update_offloading_policy(const INT4Model& model);
};

// Dynamic batch sizing based on memory availability
class DynamicBatchSizer {
public:
    DynamicBatchSizer(size_t gpu_memory_mb, size_t system_ram_mb);

    // Calculate optimal batch size
    int calculate_optimal_batch_size(const INT4Model& model,
                                   const QuantizationConfig& config,
                                   float target_memory_usage = 0.8f);

    // Adaptive batch sizing during training
    int adapt_batch_size_during_training(int current_batch_size,
                                       float current_memory_usage,
                                       float target_memory_usage);

    // Memory estimation for different batch sizes
    size_t estimate_memory_for_batch(const INT4Model& model, int batch_size,
                                   const QuantizationConfig& config);

    // Performance prediction
    float predict_performance_impact(int batch_size, const INT4Model& model);

private:
    size_t gpu_memory_mb_;
    size_t system_ram_mb_;

    // Memory modeling
    size_t estimate_model_memory(const INT4Model& model) const;
    size_t estimate_activation_memory(const INT4Model& model, int batch_size) const;
    size_t estimate_gradient_memory(const INT4Model& model, int batch_size) const;
    size_t estimate_optimizer_memory(const INT4Model& model, int batch_size) const;
};

// Comprehensive memory-efficient training manager
class MemoryEfficientTrainingManager {
public:
    MemoryEfficientTrainingManager(size_t gpu_memory_mb, size_t system_ram_mb);

    // Setup memory-efficient training
    void configure_memory_efficient_training(INT4Model& model,
                                           const QuantizationConfig& config);

    // Training step with memory optimizations
    std::vector<float> training_step_memory_efficient(
        INT4Model& model,
        const TrainingBatch& batch,
        const QuantizationConfig& config,
        int micro_batch_size = 2);

    // Memory usage monitoring
    void monitor_memory_during_training(const INT4Model& model,
                                      const TrainingBatch& batch);
    MemoryUsageReport generate_memory_report() const;

    // Automatic optimization
    void optimize_memory_usage(const INT4Model& model,
                             const TrainingBatch& batch);
    void adjust_training_parameters_for_memory(const INT4Model& model);

private:
    std::unique_ptr<GradientCheckpointingManager> checkpoint_manager_;
    std::unique_ptr<ActivationOffloadingManager> offload_manager_;
    std::unique_ptr<DynamicBatchSizer> batch_sizer_;

    // Memory optimization strategies
    void apply_gradient_checkpointing(INT4Model& model, const TrainingBatch& batch);
    void apply_activation_offloading(INT4Model& model);
    void optimize_batch_size(INT4Model& model, const TrainingBatch& batch);

    // Performance monitoring
    MemoryUsageStats current_stats_;
    void update_memory_stats();
};

// Memory usage reporting
struct MemoryUsageStats {
    size_t gpu_memory_used_mb;
    size_t cpu_memory_used_mb;
    size_t checkpoint_memory_mb;
    size_t offloaded_memory_mb;
    float memory_efficiency;  // 0.0 to 1.0
    std::chrono::system_clock::time_point timestamp;
};

struct MemoryUsageReport {
    MemoryUsageStats peak_usage;
    MemoryUsageStats average_usage;
    MemoryUsageStats current_usage;
    std::vector<std::string> optimization_recommendations;
    float training_efficiency_score;  // 0.0 to 1.0
};

} // namespace Nyx