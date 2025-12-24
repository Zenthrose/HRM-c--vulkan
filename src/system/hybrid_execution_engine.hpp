#pragma once

#include <vulkan/vulkan.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <chrono>
#include <string>
#include <cstdint>
#include <algorithm>
#include "../vulkan/quantization_types.hpp"

namespace Nyx {

// Forward declarations for components implemented elsewhere
class AdaptiveQuantizationManager;
class QuantizationEngine;

// Execution strategy for hybrid computation
enum class ExecutionStrategy {
    GPU_FP32,
    GPU_QUANTIZED,
    CPU_FALLBACK,
    HYBRID_QUANTIZED,
    UNKNOWN
};

// Performance metrics used by the engine
struct PerformanceMetrics {
    float inference_time_ms = 0.0f;
    uint32_t memory_usage_mb = 0;
    float accuracy = 0.0f;
    ExecutionStrategy strategy_used = ExecutionStrategy::UNKNOWN;
};

// Execution result for performance tracking
struct ExecutionResult {
    ExecutionStrategy execution_strategy = ExecutionStrategy::UNKNOWN;
    PrecisionLevel precision_used = PrecisionLevel::FP32;
    float inference_time_ms = 0.0f;
    uint32_t memory_usage_mb = 0;
    float accuracy_score = 0.0f;
    bool error_occurred = false;
};

// Backend performance record
struct BackendPerformance {
    double latency_ms = 0.0;
    double throughput_items_per_sec = 0.0;
    size_t memory_usage_mb = 0;
    double power_consumption_watts = 0.0;
    double utilization_percent = 0.0;
    std::chrono::system_clock::time_point measured_at;
};

// Hybrid execution engine (declarations must match the implementation)
class HybridExecutionEngine {
public:
    HybridExecutionEngine(VkDevice gpu_device, VkPhysicalDevice gpu_physical_device,
                         VkQueue gpu_queue, VkQueue cpu_queue,
                         std::shared_ptr<AdaptiveQuantizationManager> quant_manager,
                         std::shared_ptr<QuantizationEngine> quant_engine);

    ~HybridExecutionEngine();

    bool initialize();

    ExecutionResult execute_model(const Model& model, const ExecutionContext& context);

    ExecutionStrategy select_execution_strategy(const Model& model, const ExecutionContext& context);

    ExecutionResult execute_gpu_fp32(const Model& model, const ExecutionContext& context);
    ExecutionResult execute_gpu_quantized(const Model& model, const ExecutionContext& context);
    // Overloads that accept a ModelInstance so runtime can use stored per-function quantized weights
    ExecutionResult execute_gpu_fp32(const ModelInstance& instance, const ExecutionContext& context);
    // Overloads that accept a ModelInstance so runtime can use stored per-function quantized weights
    ExecutionResult execute_model(const ModelInstance& instance, const ExecutionContext& context);
    ExecutionResult execute_gpu_quantized(const ModelInstance& instance, const ExecutionContext& context);
    ExecutionResult execute_cpu_fallback(const Model& model, const ExecutionContext& context);
    ExecutionResult execute_hybrid_quantized(const Model& model, const ExecutionContext& context);
    ExecutionResult execute_hybrid_quantized(const ModelInstance& instance, const ExecutionContext& context);

    void adapt_execution_strategy(const PerformanceMetrics& metrics);
    bool switch_precision_level(ModelInstance& instance, PrecisionLevel new_precision);

    HardwareCapabilities get_hardware_capabilities() const;
    PerformanceMetrics get_performance_metrics() const;

    void enable_hybrid_execution(bool enable);
    bool is_hybrid_execution_enabled() const;

private:
    // Resource management
    void create_vulkan_resources();
    void destroy_vulkan_resources();

    // Model analysis & splitting for hybrid execution
    std::pair<Model, Model> split_model_for_hybrid_execution(const Model& model);

    // Estimators
    float estimate_performance(const Model& model, PrecisionLevel precision);
    size_t estimate_model_memory_usage(const Model& model, PrecisionLevel precision);
    float estimate_accuracy(const Model& model, PrecisionLevel precision);

    void update_performance_metrics(ExecutionStrategy strategy, const ExecutionResult& result);
    void increase_quantization_aggressiveness();
    void optimize_for_speed();

private:
    std::shared_ptr<AdaptiveQuantizationManager> quantization_manager_;
    std::shared_ptr<QuantizationEngine> quantization_engine_;

    std::unordered_map<PrecisionLevel, BackendPerformance> backend_performance_;

    VkDevice gpu_device_ = VK_NULL_HANDLE;
    VkPhysicalDevice gpu_physical_device_ = VK_NULL_HANDLE;
    VkQueue gpu_queue_ = VK_NULL_HANDLE;
    VkQueue cpu_queue_ = VK_NULL_HANDLE;
    uint32_t gpu_queue_family_ = 0;
    bool hybrid_execution_enabled_ = false;
    HardwareCapabilities hardware_caps_;
    PerformanceMetrics performance_metrics_;
    PrecisionLevel current_precision_level_ = PrecisionLevel::FP32;
};

} // namespace Nyx