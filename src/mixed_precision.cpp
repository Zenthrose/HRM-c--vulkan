#include "mixed_precision.hpp"
#include <cstring>
#include <limits>
#include <iostream>
#include <cstdint>

// FP16 Implementation
uint16_t FP16Utils::float_to_half(float value) {
    // IEEE 754 float32 to float16 conversion
    uint32_t float_bits;
    memcpy(&float_bits, &value, sizeof(float));

    uint32_t sign = (float_bits >> 31) & 0x1;
    uint32_t exponent = (float_bits >> 23) & 0xFF;
    uint32_t mantissa = float_bits & 0x7FFFFF;

    // Handle special cases
    if (exponent == 0xFF) {  // Inf or NaN
        if (mantissa != 0) return 0x7FFF;  // NaN -> NaN
        return (uint16_t)((sign << 15) | 0x7C00);  // Inf
    }

    if (exponent == 0) {  // Denormal
        // Too small for half precision
        return (uint16_t)(sign << 15);
    }

    // Normalized number
    exponent -= 127;  // Remove bias
    exponent += 15;   // Add half precision bias

    if (exponent <= 0) {
        // Underflow to zero
        return (uint16_t)(sign << 15);
    }

    if (exponent >= 31) {
        // Overflow to infinity
        return (uint16_t)((sign << 15) | 0x7C00);
    }

    // Round mantissa to 10 bits
    mantissa >>= 13;  // Keep top 10 bits of 23-bit mantissa

    return (uint16_t)((sign << 15) | (exponent << 10) | mantissa);
}

float FP16Utils::half_to_float(uint16_t value) {
    uint32_t sign = (value >> 15) & 0x1;
    uint32_t exponent = (value >> 10) & 0x1F;
    uint32_t mantissa = value & 0x3FF;

    uint32_t float_bits;

    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero
            float_bits = sign << 31;
        } else {
            // Denormal - convert to normalized float32
            exponent = 1;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            exponent += 127 - 15 - 1;
            mantissa <<= 13;
            float_bits = (sign << 31) | (exponent << 23) | mantissa;
        }
    } else if (exponent == 31) {
        // Inf or NaN
        float_bits = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    } else {
        // Normalized
        exponent += 127 - 15;
        mantissa <<= 13;
        float_bits = (sign << 31) | (exponent << 23) | mantissa;
    }

    float result;
    memcpy(&result, &float_bits, sizeof(float));
    return result;
}

std::vector<uint16_t> FP16Utils::floats_to_halfs(const std::vector<float>& values) {
    std::vector<uint16_t> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = float_to_half(values[i]);
    }
    return result;
}

std::vector<float> FP16Utils::halfs_to_floats(const std::vector<uint16_t>& values) {
    std::vector<float> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = half_to_float(values[i]);
    }
    return result;
}

uint16_t FP16Utils::add_half(uint16_t a, uint16_t b) {
    float fa = half_to_float(a);
    float fb = half_to_float(b);
    return float_to_half(fa + fb);
}

uint16_t FP16Utils::multiply_half(uint16_t a, uint16_t b) {
    float fa = half_to_float(a);
    float fb = half_to_float(b);
    return float_to_half(fa * fb);
}

bool FP16Utils::is_overflow(float value) {
    return std::abs(value) > 65504.0f;  // Max half precision value
}

bool FP16Utils::is_underflow(float value) {
    return std::abs(value) < 6.103515625e-5f && value != 0.0f;  // Min normalized half
}

// BF16 Implementation
uint16_t BF16Utils::float_to_bfloat16(float value) {
    uint32_t float_bits;
    memcpy(&float_bits, &value, sizeof(float));
    // BF16 keeps the top 8 bits of mantissa (bits 22-15 of float32)
    return (uint16_t)(float_bits >> 16);
}

float BF16Utils::bfloat16_to_float(uint16_t value) {
    uint32_t float_bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &float_bits, sizeof(float));
    return result;
}

std::vector<uint16_t> BF16Utils::floats_to_bfloats(const std::vector<float>& values) {
    std::vector<uint16_t> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = float_to_bfloat16(values[i]);
    }
    return result;
}

std::vector<float> BF16Utils::bfloats_to_floats(const std::vector<uint16_t>& values) {
    std::vector<float> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = bfloat16_to_float(values[i]);
    }
    return result;
}

// FP8 Implementation (simplified)
uint8_t FP8Utils::float_to_fp8(float value, FP8Format format) {
    // Simplified FP8 conversion (E4M3 format)
    uint32_t float_bits;
    memcpy(&float_bits, &value, sizeof(float));

    uint32_t sign = (float_bits >> 31) & 0x1;
    uint32_t exponent = (float_bits >> 23) & 0xFF;
    uint32_t mantissa = float_bits & 0x7FFFFF;

    // Convert to E4M3 format (1 sign, 4 exp, 3 mantissa)
    exponent = std::min(15u, std::max(0u, exponent - 127 + 7));  // Bias adjustment
    mantissa >>= 20;  // Keep top 3 bits

    return (uint8_t)((sign << 7) | (exponent << 3) | mantissa);
}

float FP8Utils::fp8_to_float(uint8_t value, FP8Format format) {
    uint32_t sign = (value >> 7) & 0x1;
    uint32_t exponent = (value >> 3) & 0xF;
    uint32_t mantissa = value & 0x7;

    uint32_t float_bits;

    if (exponent == 0) {
        if (mantissa == 0) {
            float_bits = sign << 31;
        } else {
            // Denormal
            exponent = 1;
            mantissa <<= 20;
            float_bits = (sign << 31) | ((127 - 7 + 1) << 23) | mantissa;
        }
    } else if (exponent == 15) {
        float_bits = (sign << 31) | (0xFF << 23) | (mantissa << 20);
    } else {
        exponent += 127 - 7;
        mantissa <<= 20;
        float_bits = (sign << 31) | (exponent << 23) | mantissa;
    }

    float result;
    memcpy(&result, &float_bits, sizeof(float));
    return result;
}

std::vector<uint8_t> FP8Utils::floats_to_fp8(const std::vector<float>& values, FP8Format format) {
    std::vector<uint8_t> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = float_to_fp8(values[i], format);
    }
    return result;
}

std::vector<float> FP8Utils::fp8_to_floats(const std::vector<uint8_t>& values, FP8Format format) {
    std::vector<float> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = fp8_to_float(values[i], format);
    }
    return result;
}

// LossScaler Implementation
LossScaler::LossScaler(float initial_scale, bool dynamic)
    : initial_scale_(initial_scale), current_scale_(initial_scale),
      dynamic_scaling_(dynamic), consecutive_good_steps_(0),
      consecutive_bad_steps_(0), growth_factor_(2.0f),
      shrink_factor_(0.5f), window_size_(2000) {}

float LossScaler::scale_loss(float loss) {
    return loss * current_scale_;
}

void LossScaler::unscale_gradients(std::vector<float>& gradients) {
    float inv_scale = 1.0f / current_scale_;
    for (auto& grad : gradients) {
        grad *= inv_scale;
    }
}

void LossScaler::update_scale(const std::vector<float>& gradients) {
    if (!dynamic_scaling_) return;

    bool has_inf_or_nan = !are_gradients_finite(gradients);

    if (has_inf_or_nan) {
        // Gradient overflow - reduce scale
        current_scale_ *= shrink_factor_;
        consecutive_good_steps_ = 0;
        consecutive_bad_steps_++;
        std::cout << "Gradient overflow detected, reducing loss scale to: " << current_scale_ << std::endl;
    } else {
        consecutive_bad_steps_ = 0;
        consecutive_good_steps_++;

        // Increase scale if we've had good steps
        if (consecutive_good_steps_ >= window_size_) {
            current_scale_ *= growth_factor_;
            consecutive_good_steps_ = 0;
            std::cout << "Increasing loss scale to: " << current_scale_ << std::endl;
        }
    }

    // Clamp scale to reasonable bounds
    current_scale_ = std::max(1.0f, std::min(current_scale_, 65536.0f * 4));
}

bool LossScaler::are_gradients_finite(const std::vector<float>& gradients) const {
    for (float grad : gradients) {
        if (!std::isfinite(grad)) {
            return false;
        }
    }
    return true;
}

// MixedPrecisionManager Implementation
MixedPrecisionManager::MixedPrecisionManager(const MixedPrecisionConfig& config)
    : config_(config), loss_scaler_(config.loss_scale, config.dynamic_loss_scaling) {}

std::vector<uint16_t> MixedPrecisionManager::convert_weights_to_half(const std::vector<float>& weights) {
    return FP16Utils::floats_to_halfs(weights);
}

std::vector<uint16_t> MixedPrecisionManager::convert_weights_to_bfloat16(const std::vector<float>& weights) {
    return BF16Utils::floats_to_bfloats(weights);
}

std::vector<uint8_t> MixedPrecisionManager::convert_weights_to_fp8(const std::vector<float>& weights) {
    return FP8Utils::floats_to_fp8(weights, FP8Utils::FP8Format::E4M3);
}

std::vector<float> MixedPrecisionManager::convert_half_to_float(const std::vector<uint16_t>& weights) {
    return FP16Utils::halfs_to_floats(weights);
}

std::vector<float> MixedPrecisionManager::convert_bfloat16_to_float(const std::vector<uint16_t>& weights) {
    return BF16Utils::bfloats_to_floats(weights);
}

std::vector<float> MixedPrecisionManager::convert_fp8_to_float(const std::vector<uint8_t>& weights) {
    return FP8Utils::fp8_to_floats(weights, FP8Utils::FP8Format::E4M3);
}

std::vector<float> MixedPrecisionManager::forward_mixed_precision(
    const std::vector<float>& input, const std::vector<uint16_t>& weights_half, bool use_fp16) {

    // Convert weights back to float for computation (in practice, this would be GPU-accelerated)
    auto weights_float = convert_half_to_float(weights_half);

    // Simplified forward pass (matrix multiplication)
    std::vector<float> output(input.size());

    // Simulate computation with potential precision loss
    for (size_t i = 0; i < input.size(); ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < weights_float.size() && j < 10; ++j) {  // Simplified
            sum += input[i] * weights_float[j];
        }
        output[i] = sum;
    }

    return output;
}

std::vector<float> MixedPrecisionManager::backward_mixed_precision(
    const std::vector<float>& gradients, float loss_scale) {

    // Unscale gradients
    std::vector<float> unscaled_gradients = gradients;
    loss_scaler_.unscale_gradients(unscaled_gradients);

    // Update loss scale based on gradient statistics
    loss_scaler_.update_scale(unscaled_gradients);

    return unscaled_gradients;
}

size_t MixedPrecisionManager::estimate_memory_savings(size_t fp32_params) const {
    size_t fp16_memory = fp32_params * 2;  // 2 bytes per parameter
    size_t fp32_memory = fp32_params * 4;  // 4 bytes per parameter

    return fp32_memory - fp16_memory;
}

std::unordered_map<std::string, float> MixedPrecisionManager::get_mixed_precision_stats() const {
    return {
        {"current_loss_scale", loss_scaler_.get_current_scale()},
        {"memory_savings_percent", 50.0f},  // FP16 saves 50% memory
        {"precision_type", config_.compute_precision == PrecisionType::FP16 ? 16.0f : 32.0f}
    };
}

// Precision-aware gradient clipping
void PrecisionAwareGradientClipper::clip_gradients_fp16(std::vector<float>& gradients, float max_norm) {
    // FP16 has limited range, so be more conservative with clipping
    float effective_max_norm = max_norm * 0.5f;  // More conservative clipping

    float total_norm = 0.0f;
    for (float grad : gradients) {
        total_norm += grad * grad;
    }
    total_norm = std::sqrt(total_norm);

    if (total_norm > effective_max_norm) {
        float scale = effective_max_norm / total_norm;
        for (float& grad : gradients) {
            grad *= scale;
        }
    }
}

void PrecisionAwareGradientClipper::clip_gradients_fp8(std::vector<float>& gradients, float max_norm) {
    // FP8 is very sensitive, use very conservative clipping
    float effective_max_norm = max_norm * 0.1f;

    float total_norm = 0.0f;
    for (float grad : gradients) {
        total_norm += grad * grad;
    }
    total_norm = std::sqrt(total_norm);

    if (total_norm > effective_max_norm) {
        float scale = effective_max_norm / total_norm;
        for (float& grad : gradients) {
            grad *= scale;
        }
    }
}

float PrecisionAwareGradientClipper::get_adaptive_clip_threshold(PrecisionType precision) {
    switch (precision) {
        case PrecisionType::FP8_E4M3:
        case PrecisionType::FP8_E5M3:
            return 0.1f;  // Very conservative for FP8
        case PrecisionType::FP16:
        case PrecisionType::BF16:
            return 1.0f;  // Standard for FP16/BF16
        case PrecisionType::FP32:
        default:
            return 5.0f;  // More aggressive for FP32
    }
}

// Benchmarking utilities
std::unordered_map<std::string, float> MixedPrecisionBenchmark::benchmark_precision_modes(
    int batch_size, int seq_len, int hidden_size) {

    std::unordered_map<std::string, float> results;

    // Simulate memory usage
    size_t params = batch_size * seq_len * hidden_size;
    results["fp32_memory_mb"] = (params * 4) / (1024.0f * 1024.0f);
    results["fp16_memory_mb"] = (params * 2) / (1024.0f * 1024.0f);
    results["fp8_memory_mb"] = (params * 1) / (1024.0f * 1024.0f);

    // Simulate speed (FP16 is typically 2x faster, FP8 potentially 4x)
    results["fp32_relative_speed"] = 1.0f;
    results["fp16_relative_speed"] = 2.0f;
    results["fp8_relative_speed"] = 4.0f;

    return results;
}

std::unordered_map<std::string, size_t> MixedPrecisionBenchmark::compare_memory_usage(size_t model_params) {
    return {
        {"fp32_bytes", model_params * 4},
        {"fp16_bytes", model_params * 2},
        {"bf16_bytes", model_params * 2},
        {"fp8_bytes", model_params * 1}
    };
}

std::unordered_map<std::string, float> MixedPrecisionBenchmark::compare_training_speed(int iterations) {
    // Simulate training loop with different precisions
    std::unordered_map<std::string, float> results;

    for (int iter = 0; iter < iterations; ++iter) {
        // FP32 baseline
        results["fp32_time"] += 1.0f;

        // FP16 is typically 1.5-2x faster
        results["fp16_time"] += 0.7f;

        // BF16 similar to FP16 but more stable
        results["bf16_time"] += 0.75f;

        // FP8 experimental, potentially 3-4x faster
        results["fp8_time"] += 0.3f;
    }

    return results;
}