#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <string>
// Bring in quantization and training batch definitions
#include "../vulkan/quantization_types.hpp"
#include "../vulkan/vulkan_trainer.hpp"

namespace Nyx {

// Forward declarations
class INT4AdamOptimizer;
class INT4GradientProcessor;
struct QuantizationParams;

#include "training_batch.hpp"

// INT4 Training Configuration
struct INT4TrainingConfig {
    int micro_batch_size = 2;           // Smaller batches for gradient accumulation
    int gradient_accumulation_steps = 4; // Accumulate gradients across micro-batches
    float gradient_clip_norm = 1.0f;    // Gradient clipping threshold
    float learning_rate = 0.001f;       // Base learning rate
    float beta1 = 0.9f;                 // Adam beta1
    float beta2 = 0.999f;               // Adam beta2
    float weight_decay = 0.01f;         // Weight decay
    bool enable_mixed_precision = true; // Use FP16 for gradients
    bool enable_gradient_checkpointing = true; // Memory optimization
    int warmup_steps = 100;             // Learning rate warmup
    int max_epochs = 100;               // Maximum training epochs
    float early_stopping_patience = 5;  // Early stopping patience
};

// INT4 Training Step Result
struct INT4TrainingStepResult {
    float loss;
    float perplexity;
    float accuracy;
    float gradient_norm;
    size_t memory_usage_mb;
    double step_time_ms;
    bool converged;
};

// INT4 Model for Training
class INT4Model {
public:
    INT4Model() = default;
    ~INT4Model() = default;

    // Model components (INT4 quantized)
    std::vector<int8_t> weights_int4;           // Packed INT4 weights
    std::vector<QuantizationParams> quant_params; // Per-layer quantization params
    std::unordered_map<std::string, int> layer_map; // Layer name to index mapping

    // Model metadata
    int num_layers;
    int hidden_size;
    int vocab_size;
    std::string model_type;

    // Utility methods
    size_t get_memory_usage_mb() const;
    bool validate_model() const;
    void save_checkpoint(const std::string& path);
    bool load_checkpoint(const std::string& path);
};

// INT4 Training Engine
class INT4TrainingEngine {
public:
    INT4TrainingEngine(std::shared_ptr<INT4AdamOptimizer> optimizer,
                      std::shared_ptr<INT4GradientProcessor> gradient_processor);

    // Training methods
    INT4TrainingStepResult train_step(INT4Model& model,
                                    const TrainingBatch& batch,
                                    const INT4TrainingConfig& config);
    // Train directly from a `ModelInstance` that may contain quantized weight blobs
    INT4TrainingStepResult train_step(INT4Model& model, ModelInstance& instance,
                                    const TrainingBatch& batch, const INT4TrainingConfig& config);

    // Gradient accumulation training
    std::vector<INT4TrainingStepResult> train_with_accumulation(
        INT4Model& model,
        const std::vector<TrainingBatch>& batches,
        const INT4TrainingConfig& config);

    // Memory-efficient training with checkpointing
    INT4TrainingStepResult train_with_checkpointing(
        INT4Model& model,
        const TrainingBatch& batch,
        const INT4TrainingConfig& config);

    // Validation and testing
    float validate_model(const INT4Model& model,
                        const std::vector<TrainingBatch>& validation_batches);

    // Learning rate scheduling
    float compute_learning_rate(int step, const INT4TrainingConfig& config);

    // Early stopping
    bool should_early_stop(const std::vector<float>& validation_losses,
                          int patience);

private:
    std::shared_ptr<INT4AdamOptimizer> optimizer_;
    std::shared_ptr<INT4GradientProcessor> gradient_processor_;

    // Training statistics
    int current_step_;
    std::vector<float> training_losses_;
    std::vector<float> validation_losses_;

    // Helper methods
    void update_training_stats(const INT4TrainingStepResult& result);
    bool check_convergence(const INT4TrainingConfig& config);
    void apply_gradient_clipping(std::vector<int8_t>& gradients, float max_norm);

    // Helpers implemented in the .cpp
    std::vector<float> convert_int4_to_fp32_gradients(const std::vector<int8_t>& int4_grads);
    float compute_loss(const INT4Model& model, const TrainingBatch& batch);
    float compute_accuracy(const INT4Model& model, const TrainingBatch& batch);
    float compute_gradient_norm(const std::vector<int8_t>& gradients);
};

// INT4 Optimizer (Adam with INT4 precision)
class INT4AdamOptimizer {
public:
    INT4AdamOptimizer(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f);

    // Initialize optimizer state for INT4 model
    void initialize_state(const INT4Model& model);

    // Update weights using INT4 gradients
    void step(INT4Model& model,
             const std::vector<int8_t>& gradients_int4,
             const std::vector<QuantizationParams>& grad_params,
             float learning_rate);

    // Mixed precision step (INT4 weights, FP16 gradients)
    void step_mixed_precision(INT4Model& model,
                            const std::vector<float>& gradients_fp16,
                            float learning_rate);

    // Learning rate adaptation
    void adapt_learning_rate(float current_loss, float target_loss);

private:
    float lr_;
    float beta1_, beta2_;
    float eps_ = 1e-8f;
    int step_count_;

    // Optimizer state (packed INT4 for memory efficiency)
    std::vector<int8_t> m_int4_;  // First moment estimate
    std::vector<int8_t> v_int4_;  // Second moment estimate
    std::vector<QuantizationParams> m_params_, v_params_;
};

// INT4 Gradient Processor
class INT4GradientProcessor {
public:
    INT4GradientProcessor();

    // Compute gradients in INT4 precision
    std::vector<int8_t> compute_gradients_int4(const INT4Model& model,
                                             const TrainingBatch& batch,
                                             const QuantizationParams& output_params);

    // Accumulate gradients across micro-batches
    void accumulate_gradients(std::vector<int8_t>& accumulated_grads,
                            const std::vector<int8_t>& batch_grads,
                            int micro_batch_size);

    // Convert gradients between precisions
    std::vector<float> int4_to_fp32_gradients(const std::vector<int8_t>& grads_int4,
                                            const QuantizationParams& params);

    std::vector<int8_t> fp32_to_int4_gradients(const std::vector<float>& grads_fp32,
                                            const QuantizationParams& params);

    // Gradient statistics and monitoring
    float compute_gradient_norm(const std::vector<int8_t>& grads_int4,
                              const QuantizationParams& params);

    bool detect_gradient_vanishing(const std::vector<int8_t>& grads_int4,
                                 const QuantizationParams& params);

    bool detect_gradient_exploding(const std::vector<int8_t>& grads_int4,
                                 const QuantizationParams& params);

private:
    // Gradient computation helpers
    std::vector<int8_t> compute_cross_entropy_gradients(const INT4Model& model,
                                                     const TrainingBatch& batch);

    std::vector<int8_t> compute_transformer_gradients(const INT4Model& model,
                                                   const TrainingBatch& batch);
};

} // namespace Nyx