#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdint>

/**
 * Mixed Precision Training Support
 *
 * Implements FP16, BF16, and FP8 training for 2x memory savings
 * and potential speedup with proper hardware support.
 */

enum class PrecisionType {
    FP32,    // Standard 32-bit float
    FP16,    // 16-bit float (half precision)
    BF16,    // Brain float 16 (better dynamic range)
    FP8_E4M3, // 8-bit float (experimental)
    FP8_E5M3  // 8-bit float with extended range
};

struct MixedPrecisionConfig {
    bool enabled = true;
    PrecisionType compute_precision = PrecisionType::FP16;
    PrecisionType memory_precision = PrecisionType::FP16;
    bool use_gradient_scaling = true;
    float loss_scale = 65536.0f;
    bool dynamic_loss_scaling = true;
    float loss_scale_growth_factor = 2.0f;
    float loss_scale_shrink_factor = 0.5f;
    int loss_scale_window = 2000;
    bool use_bfloat16 = false;  // Alternative to FP16
    bool enable_fp8 = false;    // Experimental FP8 support
};

/**
 * Half-precision (FP16) utilities
 */
class FP16Utils {
public:
    // Convert FP32 to FP16
    static uint16_t float_to_half(float value);

    // Convert FP16 to FP32
    static float half_to_float(uint16_t value);

    // Vectorized conversions
    static std::vector<uint16_t> floats_to_halfs(const std::vector<float>& values);
    static std::vector<float> halfs_to_floats(const std::vector<uint16_t>& values);

    // Arithmetic operations in FP16
    static uint16_t add_half(uint16_t a, uint16_t b);
    static uint16_t multiply_half(uint16_t a, uint16_t b);

    // Check for overflow/underflow
    static bool is_overflow(float value);
    static bool is_underflow(float value);
};

/**
 * BFloat16 utilities (better dynamic range than FP16)
 */
class BF16Utils {
public:
    // Convert FP32 to BF16
    static uint16_t float_to_bfloat16(float value);

    // Convert BF16 to FP32
    static float bfloat16_to_float(uint16_t value);

    // Vectorized conversions
    static std::vector<uint16_t> floats_to_bfloats(const std::vector<float>& values);
    static std::vector<float> bfloats_to_floats(const std::vector<uint16_t>& values);
};

/**
 * FP8 utilities (experimental 8-bit precision)
 */
class FP8Utils {
public:
    enum class FP8Format { E4M3, E5M3 };

    // Convert FP32 to FP8
    static uint8_t float_to_fp8(float value, FP8Format format = FP8Format::E4M3);

    // Convert FP8 to FP32
    static float fp8_to_float(uint8_t value, FP8Format format = FP8Format::E4M3);

    // Vectorized conversions
    static std::vector<uint8_t> floats_to_fp8(const std::vector<float>& values, FP8Format format = FP8Format::E4M3);
    static std::vector<float> fp8_to_floats(const std::vector<uint8_t>& values, FP8Format format = FP8Format::E4M3);
};

/**
 * Loss scaling for mixed precision training
 */
class LossScaler {
public:
    LossScaler(float initial_scale = 65536.0f, bool dynamic = true);

    // Scale loss for backward pass
    float scale_loss(float loss);

    // Unscale gradients
    void unscale_gradients(std::vector<float>& gradients);

    // Update scale based on gradient statistics
    void update_scale(const std::vector<float>& gradients);

    // Check if gradients are finite
    bool are_gradients_finite(const std::vector<float>& gradients) const;

    // Get current scale
    float get_current_scale() const { return current_scale_; }

    // Reset scale
    void reset_scale() { current_scale_ = initial_scale_; }

private:
    float initial_scale_;
    float current_scale_;
    bool dynamic_scaling_;
    int consecutive_good_steps_;
    int consecutive_bad_steps_;
    float growth_factor_;
    float shrink_factor_;
    int window_size_;
};

/**
 * Mixed Precision Manager
 */
class MixedPrecisionManager {
public:
    MixedPrecisionManager(const MixedPrecisionConfig& config);

    // Convert model weights to target precision
    std::vector<uint16_t> convert_weights_to_half(const std::vector<float>& weights);
    std::vector<uint16_t> convert_weights_to_bfloat16(const std::vector<float>& weights);
    std::vector<uint8_t> convert_weights_to_fp8(const std::vector<float>& weights);

    // Convert back to FP32 for operations
    std::vector<float> convert_half_to_float(const std::vector<uint16_t>& weights);
    std::vector<float> convert_bfloat16_to_float(const std::vector<uint16_t>& weights);
    std::vector<float> convert_fp8_to_float(const std::vector<uint8_t>& weights);

    // Forward pass in mixed precision
    std::vector<float> forward_mixed_precision(
        const std::vector<float>& input,
        const std::vector<uint16_t>& weights_half,
        bool use_fp16 = true);

    // Backward pass with gradient scaling
    std::vector<float> backward_mixed_precision(
        const std::vector<float>& gradients,
        float loss_scale);

    // Memory usage estimation
    size_t estimate_memory_savings(size_t fp32_params) const;

    // Performance monitoring
    std::unordered_map<std::string, float> get_mixed_precision_stats() const;

private:
    MixedPrecisionConfig config_;
    LossScaler loss_scaler_;
    std::unordered_map<std::string, float> stats_;

    // Internal conversion buffers
    std::vector<uint16_t> half_buffer_;
    std::vector<uint16_t> bfloat_buffer_;
    std::vector<uint8_t> fp8_buffer_;
};

/**
 * Automatic Mixed Precision (AMP) utilities
 */
class AutomaticMixedPrecision {
public:
    struct AMPConfig {
        bool enabled = true;
        PrecisionType opt_level = PrecisionType::FP16;  // O1 = FP16, O2 = FP16 + more aggressive
        bool keep_batchnorm_fp32 = true;
        bool cast_model_outputs = true;
        std::vector<std::string> fp32_modules;  // Modules to keep in FP32
    };

    AutomaticMixedPrecision(const AMPConfig& config);

    // Automatically wrap model for mixed precision
    template<typename ModelType>
    void wrap_model(ModelType& model);

    // Scale loss for mixed precision training
    float scale_loss(float loss);

    // Backward pass with automatic gradient scaling
    void backward_scaled(const std::vector<float>& loss_gradients);

    // Update optimizer with scaled gradients
    void step_optimizer_scaled();

    // Check for gradient overflow
    bool check_gradient_overflow(const std::vector<float>& gradients);

private:
    AMPConfig config_;
    LossScaler scaler_;
    bool gradient_overflow_;
};

/**
 * Precision-aware gradient clipping
 */
class PrecisionAwareGradientClipper {
public:
    // Clip gradients based on precision type
    static void clip_gradients_fp16(std::vector<float>& gradients, float max_norm);
    static void clip_gradients_fp8(std::vector<float>& gradients, float max_norm);

    // Adaptive clipping based on precision
    static float get_adaptive_clip_threshold(PrecisionType precision);
};

/**
 * Mixed precision benchmarking utilities
 */
class MixedPrecisionBenchmark {
public:
    // Benchmark different precision modes
    static std::unordered_map<std::string, float> benchmark_precision_modes(
        int batch_size, int seq_len, int hidden_size);

    // Memory usage comparison
    static std::unordered_map<std::string, size_t> compare_memory_usage(
        size_t model_params);

    // Speed comparison
    static std::unordered_map<std::string, float> compare_training_speed(
        int iterations = 1000);
};