#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <vector>
#include <algorithm>

#ifndef NO_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace Nyx {

// Bring fixed-width integer types into the Nyx namespace and define float32_t
using uint8_t = std::uint8_t;
using int8_t  = std::int8_t;
using uint16_t = std::uint16_t;
using uint32_t = std::uint32_t;
using uint64_t = std::uint64_t;
using float32_t = float;

// Precision levels for quantization
enum class PrecisionLevel {
    FP32,     // Full precision (32-bit float)
    FP16,     // Half precision (16-bit float)
    BF16,     // BFloat16 (16-bit float)
    INT8,     // 8-bit integer
    INT4,     // 4-bit integer
    DYNAMIC,  // Runtime precision selection
    MIXED     // Different precisions for different components
};

// Quantization configuration
struct QuantizationConfig {
    PrecisionLevel precision_level;
    bool per_channel_quantization;
    float32_t calibration_factor;
};

// Provide a simple ordering for QuantizationConfig so it can be used in std::pair comparisons
inline bool operator<(const QuantizationConfig& a, const QuantizationConfig& b) {
    if (a.precision_level != b.precision_level) return a.precision_level < b.precision_level;
    if (a.per_channel_quantization != b.per_channel_quantization) return a.per_channel_quantization < b.per_channel_quantization;
    return a.calibration_factor < b.calibration_factor;
}

// Quantization parameters
struct QuantizationParams {
    float32_t scale;
    int8_t zero_point;
    PrecisionLevel precision;
};

// Hardware capabilities
struct HardwareCapabilities {
    uint32_t gpu_memory_mb;
    uint32_t cpu_memory_mb;
    bool has_tensor_cores;
    bool supports_fp16;
    bool supports_int8;
    bool supports_int4;
    
    enum class PerformanceTier {
        ULTRA_LOW, LOW, MEDIUM, HIGH, ULTRA_HIGH
    } performance_tier;
};

// Task requirements
struct TaskRequirements {
    PrecisionLevel min_precision;
    PrecisionLevel max_precision;
    uint32_t max_memory_mb;
    float32_t min_accuracy;
};

// Full type definitions to avoid incomplete type issues

// Tensor structure
typedef struct {
    float32_t* data;
    uint32_t* shape;
    uint32_t shape_size;
} Tensor;

// Model structure
typedef struct {
    struct WeightMapEntry {
        const char* name;
        float32_t* weights;
        uint32_t size;
    }* weight_maps;
    uint32_t num_weights;
} Model;

// Quantized model structure
struct QuantizedModel {
    uint8_t* weights = nullptr;
    std::unique_ptr<QuantizationParams> quantization_params;
    PrecisionLevel precision = PrecisionLevel::FP32;
    uint32_t weights_size = 0;
#ifndef NO_VULKAN
    // GPU-side buffer for dequantized weights (for compute kernels)
    VkBuffer weight_buffer = VK_NULL_HANDLE;
    VkDeviceMemory weight_memory = VK_NULL_HANDLE;
#endif

    QuantizedModel() = default;

    // Non-copyable to avoid accidental double-free
    QuantizedModel(const QuantizedModel&) = delete;
    QuantizedModel& operator=(const QuantizedModel&) = delete;

    // Movable
    QuantizedModel(QuantizedModel&& other) noexcept {
        weights = other.weights;
        quantization_params = std::move(other.quantization_params);
        precision = other.precision;
        weights_size = other.weights_size;
        other.weights = nullptr;
        other.weights_size = 0;
        other.precision = PrecisionLevel::FP32;
    }

    QuantizedModel& operator=(QuantizedModel&& other) noexcept {
        if (this != &other) {
            // free existing
            if (weights) delete[] weights;
            weights = other.weights;
            quantization_params = std::move(other.quantization_params);
            precision = other.precision;
            weights_size = other.weights_size;
            other.weights = nullptr;
            other.weights_size = 0;
            other.precision = PrecisionLevel::FP32;
        }
        return *this;
    }

    ~QuantizedModel() {
        if (weights) {
            delete[] weights;
            weights = nullptr;
        }
    }
};

// Model instance structure
typedef struct {
    Model model;
    PrecisionLevel current_precision;
    Tensor* quantized_weights;
    uint32_t num_quantized_weights;
    // Per-function / per-weight-map quantized representations (keyed by weight map name or function id)
    std::unordered_map<std::string, QuantizedModel> function_quantized_models;
} ModelInstance;

// Runtime context structure
typedef struct {
    float32_t current_accuracy;
    uint32_t current_memory_usage;
    float32_t performance_score;
} RuntimeContext;

// Add TaskRequirements to runtime context so callers can request resources
struct RuntimeContextWithRequirements : public RuntimeContext {
    TaskRequirements requirements; // desired/observed task requirements
};

// Keep an alias ExecutionContext pointing at the complete runtime context
using ExecutionContext = RuntimeContextWithRequirements;

// Helper: convert PrecisionLevel to a string (simple numeric fallback)
inline std::string precision_level_to_string(PrecisionLevel p) {
    return std::to_string(static_cast<int>(p));
}

// Quantization performance structure
typedef struct {
    PrecisionLevel precision;
    float32_t accuracy_score;
    uint32_t memory_usage_mb;
    float32_t inference_speed_ms;
    uint64_t measured_at;
    const char* hardware_profile;
} QuantizationPerformance;

// Experiment database structure
typedef struct {
    QuantizationPerformance* experiments;
    uint32_t num_experiments;
} ExperimentDatabase;

} // namespace Nyx