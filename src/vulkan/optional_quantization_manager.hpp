#pragma once

// Forward declarations to avoid circular dependencies
namespace Nyx {
enum class PrecisionLevel;
struct QuantizationConfig;
struct HardwareCapabilities;
struct TaskRequirements;
struct QuantizationPerformance;
class Model;
}

// QUANTIZATION_COMPONENT: Optional Quantization Manager - Nyx can modify or remove
// This is the SAFE WRAPPER that ensures Nyx functions normally even if quantization fails
// Fallback: All operations fall back to standard FP32 behavior

namespace Nyx {

/**
 * Safe wrapper for all quantization functionality
 * Ensures Nyx continues working normally even if quantization is disabled or fails
 */
class OptionalQuantizationManager {
public:
    OptionalQuantizationManager();
    ~OptionalQuantizationManager();

    // Initialization with safety checks
    bool initialize(const HardwareCapabilities& hw_caps);
    void shutdown();

    // Safe quantization operations - all have FP32 fallbacks
    bool quantize_model_safe(Model& model, const QuantizationConfig& config);
    bool dequantize_model_safe(Model& model);
    float get_quantized_accuracy_estimate(const Model& model, PrecisionLevel precision);

    // Training integration - safe fallbacks
    bool enable_quantized_training();
    bool train_with_quantization_fallback(const std::string& training_data_path);

    // Runtime precision management
    bool switch_precision_safely(PrecisionLevel new_precision);
    PrecisionLevel get_current_precision() const;
    bool is_quantization_available() const;

    // Resource management
    size_t get_memory_usage() const;
    bool should_use_quantization(const TaskRequirements& requirements) const;

    // Self-modification support
    std::vector<std::string> get_quantization_components() const;
    bool disable_quantization_component(const std::string& component_name);
    bool enable_quantization_component(const std::string& component_name);

    // Health monitoring
    bool is_system_healthy() const;
    std::string get_health_report() const;
    void reset_to_safe_state();

private:
    // Component pointers - may be null if quantization unavailable
    std::shared_ptr<QuantizationEngine> quantization_engine_;
    std::shared_ptr<AdaptiveQuantizationManager> adaptive_manager_;
    std::shared_ptr<INT4TrainingEngine> training_engine_;
    std::shared_ptr<HybridExecutionEngine> execution_engine_;
    std::shared_ptr<MetaQuantizationLearner> meta_learner_;

    // State tracking
    bool initialized_;
    bool quantization_available_;
    PrecisionLevel current_precision_;
    HardwareCapabilities hardware_caps_;

    // Fallback mechanisms
    bool use_cpu_fallback_;
    bool aggressive_memory_management_;

    // Health monitoring
    int consecutive_failures_;
    std::vector<std::string> error_history_;
    std::unordered_map<std::string, bool> component_status_;

    // Private helper methods
    bool detect_hardware_capabilities();
    bool initialize_quantization_components();
    void setup_fallback_mechanisms();
    bool validate_quantization_safety();
    void log_quantization_event(const std::string& event, bool success);
    bool attempt_recovery_from_failure();

    // Component management
    bool safe_component_initialization(std::function<bool()> init_func,
                                    const std::string& component_name);
    void mark_component_failed(const std::string& component_name);
    void mark_component_recovered(const std::string& component_name);

    // Memory and safety guards
    bool check_memory_safety(size_t required_mb);
    bool validate_precision_switch(PrecisionLevel from, PrecisionLevel to);
    void emergency_cleanup();
};

// Global safe quantization interface
class QuantizationSafetyWrapper {
public:
    static QuantizationSafetyWrapper& getInstance();

    // Safe global operations
    bool safe_quantize_operation(std::function<bool()> operation,
                               const std::string& operation_name);

    bool is_quantization_safe() const;
    void disable_all_quantization();
    void enable_safe_quantization();

    // Emergency recovery
    void emergency_reset();
    std::string get_safety_report() const;

private:
    QuantizationSafetyWrapper();
    ~QuantizationSafetyWrapper();

    bool safety_enabled_;
    int global_failure_count_;
    std::vector<std::string> safety_log_;

    // Singleton pattern
    static std::unique_ptr<QuantizationSafetyWrapper> instance_;
    static bool initialized_;
};

} // namespace Nyx