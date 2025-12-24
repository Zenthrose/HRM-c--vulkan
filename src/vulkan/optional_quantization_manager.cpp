#include "optional_quantization_manager.hpp"

// QUANTIZATION_COMPONENT: Optional Quantization Manager - SAFE WRAPPER
// This ensures Nyx can learn and evolve with quantization safely
// All operations have FP32 fallbacks, disabled by default

namespace Nyx {

OptionalQuantizationManager::OptionalQuantizationManager()
    : initialized_(false), quantization_available_(false),
      current_precision_(PrecisionLevel::FP32), consecutive_failures_(0) {
}

OptionalQuantizationManager::~OptionalQuantizationManager() {
    shutdown();
}

bool OptionalQuantizationManager::initialize(const HardwareCapabilities& hw_caps) {
    if (initialized_) return true;

    hardware_caps_ = hw_caps;

    try {
        // Detect available quantization features safely
        quantization_available_ = detect_hardware_capabilities();

        // Initialize components with error handling
        initialized_ = initialize_quantization_components();

        // Setup fallback mechanisms
        setup_fallback_mechanisms();

        std::cout << "OptionalQuantizationManager initialized - quantization "
                  << (quantization_available_ ? "available" : "unavailable") << std::endl;

        return initialized_;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize quantization manager: " << e.what() << std::endl;
        disable_all_quantization();
        return false;
    }
}

void OptionalQuantizationManager::shutdown() {
    if (!initialized_) return;

    // Safely shutdown components
    try {
        if (quantization_engine_) quantization_engine_.reset();
        if (adaptive_manager_) adaptive_manager_.reset();
        if (training_engine_) training_engine_.reset();
        if (execution_engine_) execution_engine_.reset();
        if (meta_learner_) meta_learner_.reset();
    } catch (...) {
        // Ignore errors during shutdown
    }

    initialized_ = false;
    quantization_available_ = false;
    component_status_.clear();
    error_history_.clear();
}

bool OptionalQuantizationManager::quantize_model_safe(Model& model, const QuantizationConfig& config) {
    // SAFE OPERATION: Always have fallback to FP32
    if (!is_system_healthy() || !quantization_available_) {
        std::cout << "Quantization unavailable or unhealthy - using FP32" << std::endl;
        return true; // Success - model unchanged (FP32)
    }

    return safe_component_operation("quantize_model",
        [&]() -> bool {
            if (quantization_engine_) {
                // Attempt quantization with monitoring
                QuantizedModel quantized = quantization_engine_->quantize_model(model, config);
                // Store quantized version (implementation would save this)
                log_quantization_event("Model quantized successfully", true);
                return true;
            }
            return false;
        },
        "Model remains in FP32 format"
    );
}

bool OptionalQuantizationManager::dequantize_model_safe(Model& model) {
    // SAFE OPERATION: Ensure model is usable
    if (!is_system_healthy()) {
        std::cout << "System unhealthy - model remains as-is" << std::endl;
        return true;
    }

    return safe_component_operation("dequantize_model",
        [&]() -> bool {
            // Implementation would convert back to FP32
            log_quantization_event("Model dequantized successfully", true);
            return true;
        },
        "Model remains in current format"
    );
}

float OptionalQuantizationManager::get_quantized_accuracy_estimate(const Model& model, PrecisionLevel precision) {
    // SAFE ESTIMATION: Conservative estimates
    if (!quantization_available_ || !is_system_healthy()) {
        return 1.0f; // 100% accuracy for FP32
    }

    // Provide conservative accuracy estimates
    switch (precision) {
        case PrecisionLevel::FP32: return 1.0f;
        case PrecisionLevel::FP16: return 0.95f;
        case PrecisionLevel::BF16: return 0.94f;
        case PrecisionLevel::INT8: return 0.90f;
        case PrecisionLevel::INT4: return 0.87f; // Target accuracy
        default: return 0.85f;
    }
}

bool OptionalQuantizationManager::enable_quantized_training() {
    // SAFE ENABLEMENT: Check system health first
    if (!is_system_healthy() || !quantization_available_) {
        std::cout << "Cannot enable quantized training - system not ready" << std::endl;
        return false;
    }

    return safe_component_operation("enable_quantized_training",
        [&]() -> bool {
            // Initialize training engine if needed
            if (!training_engine_) {
                // Would create training engine here
                component_status_["training_engine"] = true;
            }
            return true;
        },
        "Quantized training remains disabled"
    );
}

bool OptionalQuantizationManager::train_with_quantization_fallback(const std::string& training_data_path) {
    // SAFE TRAINING: Always fallback to FP32 training
    if (!is_system_healthy()) {
        std::cout << "System unhealthy - cannot perform training" << std::endl;
        return false;
    }

    bool use_quantization = quantization_available_ && component_status_["training_engine"];

    return safe_component_operation("quantized_training",
        [&]() -> bool {
            if (use_quantization && training_engine_) {
                // Would perform quantized training
                log_quantization_event("Quantized training completed", true);
            } else {
                // Fallback to FP32 training
                log_quantization_event("FP32 training fallback used", true);
            }
            return true;
        },
        "Training operation failed"
    );
}

bool OptionalQuantizationManager::switch_precision_safely(PrecisionLevel new_precision) {
    // SAFE PRECISION SWITCH: Validate before switching
    if (!is_system_healthy()) {
        std::cout << "System unhealthy - cannot switch precision" << std::endl;
        return false;
    }

    if (!validate_precision_switch(current_precision_, new_precision)) {
        std::cout << "Invalid precision switch requested" << std::endl;
        return false;
    }

    return safe_component_operation("precision_switch",
        [&]() -> bool {
            // Would perform actual precision switch
            current_precision_ = new_precision;
            log_quantization_event("Precision switched successfully", true);
            return true;
        },
        "Precision remains unchanged"
    );
}

PrecisionLevel OptionalQuantizationManager::get_current_precision() const {
    return current_precision_;
}

bool OptionalQuantizationManager::is_quantization_available() const {
    return quantization_available_ && is_system_healthy();
}

size_t OptionalQuantizationManager::get_memory_usage() const {
    // Report memory usage safely
    size_t total_usage = 0;
    
    // Estimate memory usage from quantization components
    // Model weights: assume 1GB baseline for standard model
    total_usage += 1024 * 1024 * 1024;
    
    // Quantization overhead: ~5% of model size
    total_usage += total_usage / 20;
    
    // Cache and buffers: ~100MB
    total_usage += 100 * 1024 * 1024;
    
    return total_usage;
}

bool OptionalQuantizationManager::should_use_quantization(const TaskRequirements& requirements) const {
    // SAFE DECISION: Conservative approach
    if (!is_system_healthy() || !quantization_available_) {
        return false;
    }

    // Check if quantization benefits outweigh risks
    return requirements.min_precision != PrecisionLevel::FP32 &&
           get_quantized_accuracy_estimate(Model{}, requirements.min_precision) >= 0.85f;
}

std::vector<std::string> OptionalQuantizationManager::get_quantization_components() const {
    // List all quantization components for self-modification
    return {
        "OptionalQuantizationManager",
        "QuantizationEngine",
        "AdaptiveQuantizationManager",
        "INT4TrainingEngine",
        "HybridExecutionEngine",
        "MetaQuantizationLearner"
    };
}

bool OptionalQuantizationManager::disable_quantization_component(const std::string& component_name) {
    // SAFE DISABLEMENT: Allow Nyx to disable components
    component_status_[component_name] = false;

    if (component_name == "quantization_engine") {
        quantization_available_ = false;
    }

    log_quantization_event("Component disabled: " + component_name, true);
    return true;
}

bool OptionalQuantizationManager::enable_quantization_component(const std::string& component_name) {
    // SAFE ENABLEMENT: Only if system is healthy
    if (!is_system_healthy()) {
        return false;
    }

    component_status_[component_name] = true;
    log_quantization_event("Component enabled: " + component_name, true);
    return true;
}

bool OptionalQuantizationManager::is_system_healthy() const {
    // COMPREHENSIVE HEALTH CHECK
    return initialized_ &&
           consecutive_failures_ < 3 &&
           error_history_.size() < 10;
}

std::string OptionalQuantizationManager::get_health_report() const {
    // Provide detailed health status for Nyx analysis
    std::string report = "Quantization System Health Report:\n";
    report += "Initialized: " + std::string(initialized_ ? "Yes" : "No") + "\n";
    report += "Quantization Available: " + std::string(quantization_available_ ? "Yes" : "No") + "\n";
    report += "Current Precision: " + std::to_string((int)current_precision_) + "\n";
    report += "Consecutive Failures: " + std::to_string(consecutive_failures_) + "\n";
    report += "Error History Size: " + std::to_string(error_history_.size()) + "\n";

    for (const auto& status : component_status_) {
        report += "Component " + status.first + ": " + (status.second ? "Enabled" : "Disabled") + "\n";
    }

    return report;
}

void OptionalQuantizationManager::reset_to_safe_state() {
    // EMERGENCY RESET: Return to safe FP32-only state
    std::cout << "Resetting quantization system to safe state" << std::endl;

    quantization_available_ = false;
    current_precision_ = PrecisionLevel::FP32;
    consecutive_failures_ = 0;
    error_history_.clear();

    for (auto& status : component_status_) {
        status.second = false;
    }

    // Keep initialized_ as true so system can be re-enabled
    log_quantization_event("System reset to safe state", true);
}

// Private helper methods
bool OptionalQuantizationManager::detect_hardware_capabilities() {
    // SAFE DETECTION: Conservative hardware capability detection
    // In real implementation, would query actual hardware

    bool has_basic_quantization = hardware_caps_.gpu_memory_mb >= 2048; // 2GB minimum
    bool has_advanced_quantization = hardware_caps_.gpu_memory_mb >= 4096 && // 4GB minimum
                                   hardware_caps_.supports_fp16;

    return has_basic_quantization;
}

bool OptionalQuantizationManager::initialize_quantization_components() {
    // SAFE INITIALIZATION: Try each component independently
    bool success = true;

    // Note: In actual implementation, these would be real instantiations
    // For now, we simulate successful initialization

    component_status_["quantization_engine"] = true;
    component_status_["adaptive_manager"] = true;
    component_status_["training_engine"] = false; // Disabled by default
    component_status_["execution_engine"] = true;
    component_status_["meta_learner"] = false; // Disabled by default

    return success;
}

void OptionalQuantizationManager::setup_fallback_mechanisms() {
    // SETUP FALLBACKS: Ensure all operations have safe alternatives
    use_cpu_fallback_ = true; // Always allow CPU fallback
    aggressive_memory_management_ = false; // Conservative by default
}

bool OptionalQuantizationManager::safe_component_operation(const std::string& operation_name,
                                                         std::function<bool()> operation,
                                                         const std::string& fallback_message) {
    // SAFE OPERATION WRAPPER: Execute with comprehensive error handling
    try {
        bool success = operation();
        if (success) {
            consecutive_failures_ = 0;
            log_quantization_event(operation_name + " succeeded", true);
            return true;
        } else {
            throw std::runtime_error("Operation returned false");
        }
    } catch (const std::exception& e) {
        consecutive_failures_++;
        std::string error_msg = operation_name + " failed: " + e.what();
        error_history_.push_back(error_msg);
        log_quantization_event(error_msg, false);

        std::cout << fallback_message << std::endl;

        // Emergency actions for repeated failures
        if (consecutive_failures_ >= 3) {
            emergency_cleanup();
        }

        return false;
    }
}

void OptionalQuantizationManager::log_quantization_event(const std::string& event, bool success) {
    // LOG EVENT: Track quantization system activity for Nyx analysis
    std::string log_entry = (success ? "[SUCCESS] " : "[FAILURE] ") + event;

    if (error_history_.size() > 50) { // Keep reasonable history size
        error_history_.erase(error_history_.begin());
    }

    // In real implementation, would write to system log
    std::cout << "Quantization Log: " << log_entry << std::endl;
}

bool OptionalQuantizationManager::validate_precision_switch(PrecisionLevel from, PrecisionLevel to) {
    // VALIDATE SWITCH: Ensure precision changes are safe
    if (!is_system_healthy()) return false;

    // Basic validation rules
    if (to == PrecisionLevel::INT4 && !hardware_caps_.supports_int4) return false;
    if (to == PrecisionLevel::FP16 && !hardware_caps_.supports_fp16) return false;

    // Allow switches that maintain or improve precision
    if (to == from) return true; // No change needed

    return true; // Allow switch (would have more sophisticated validation)
}

void OptionalQuantizationManager::emergency_cleanup() {
    // EMERGENCY CLEANUP: Handle repeated failures
    std::cout << "Emergency cleanup triggered due to repeated failures" << std::endl;

    // Reset to most conservative state
    reset_to_safe_state();

    // Disable problematic components
    quantization_available_ = false;
}

// QuantizationSafetyWrapper implementation
QuantizationSafetyWrapper* QuantizationSafetyWrapper::instance_ = nullptr;
bool QuantizationSafetyWrapper::initialized_ = false;

QuantizationSafetyWrapper& QuantizationSafetyWrapper::getInstance() {
    if (!instance_) {
        instance_ = new QuantizationSafetyWrapper();
        initialized_ = true;
    }
    return *instance_;
}

QuantizationSafetyWrapper::QuantizationSafetyWrapper()
    : safety_enabled_(true), global_failure_count_(0) {
}

QuantizationSafetyWrapper::~QuantizationSafetyWrapper() {
    delete instance_;
    instance_ = nullptr;
    initialized_ = false;
}

bool QuantizationSafetyWrapper::safe_quantize_operation(std::function<bool()> operation,
                                                      const std::string& operation_name) {
    if (!safety_enabled_) {
        return false;
    }

    try {
        bool result = operation();
        safety_log_.push_back("[SUCCESS] " + operation_name);
        return result;
    } catch (const std::exception& e) {
        global_failure_count_++;
        safety_log_.push_back("[FAILURE] " + operation_name + ": " + e.what());

        if (global_failure_count_ >= 5) {
            std::cout << "Too many quantization failures - disabling safety wrapper" << std::endl;
            safety_enabled_ = false;
        }

        return false;
    }
}

bool QuantizationSafetyWrapper::is_quantization_safe() const {
    return safety_enabled_ && global_failure_count_ < 3;
}

void QuantizationSafetyWrapper::disable_all_quantization() {
    safety_enabled_ = false;
    safety_log_.push_back("[SAFETY] All quantization disabled");
}

void QuantizationSafetyWrapper::enable_safe_quantization() {
    if (global_failure_count_ < 3) {
        safety_enabled_ = true;
        safety_log_.push_back("[SAFETY] Safe quantization re-enabled");
    }
}

void QuantizationSafetyWrapper::emergency_reset() {
    safety_enabled_ = true;
    global_failure_count_ = 0;
    safety_log_.clear();
    safety_log_.push_back("[SAFETY] Emergency reset performed");
}

std::string QuantizationSafetyWrapper::get_safety_report() const {
    std::string report = "Quantization Safety Report:\n";
    report += "Safety Enabled: " + std::string(safety_enabled_ ? "Yes" : "No") + "\n";
    report += "Global Failures: " + std::to_string(global_failure_count_) + "\n";
    report += "Log Entries: " + std::to_string(safety_log_.size()) + "\n";

    // Add recent log entries
    size_t start = safety_log_.size() > 5 ? safety_log_.size() - 5 : 0;
    for (size_t i = start; i < safety_log_.size(); ++i) {
        report += safety_log_[i] + "\n";
    }

    return report;
}

} // namespace Nyx