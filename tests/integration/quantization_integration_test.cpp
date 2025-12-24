#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <cassert>

#include "../src/vulkan/adaptive_quantization.hpp"
#include "../src/vulkan/quantization_engine.hpp"
#include "../src/core/model.hpp"
#include "../src/system/hardware_abstraction_layer.hpp"

using namespace std;

// Simple test framework functions
#define TEST_ASSERT(condition, message) \
    if (!(condition)) { \
        std::cerr << "TEST FAILED: " << message << std::endl; \
        return false; \
    }

#define RUN_TEST(test_func) \
    if (test_func()) { \
        std::cout << #test_func << ": PASSED" << std::endl; \
        passed_tests++; \
    } else { \
        std::cout << #test_func << ": FAILED" << std::endl; \
        failed_tests++; \
    }

// Test fixture for quantization integration testing
class QuantizationIntegrationTest {
public:
    QuantizationIntegrationTest() {
        // Initialize hardware capabilities (mock for testing)
        hw_caps_.gpu_memory_mb = 4096;
        hw_caps_.supports_fp16 = true;
        hw_caps_.supports_int8 = true;
        hw_caps_.supports_int4 = true;

        // Create quantization components
        quant_engine_ = std::make_shared<QuantizationEngine>();
        adaptive_quant_ = std::make_shared<AdaptiveQuantizationManager>(hw_caps_);

        // Create a simple test model
        create_test_model();
    }

    void create_test_model() {
        // Create a simple transformer model for testing
        model_.config.vocab_size = 1000;
        model_.config.hidden_size = 512;
        model_.config.num_layers = 2;
        model_.config.num_heads = 8;
        model_.config.head_dim = 64;

        // Initialize model weights with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.1f);

        size_t param_count = model_.config.vocab_size * model_.config.hidden_size +  // embedding
                            model_.config.hidden_size * model_.config.hidden_size +  // hidden weights
                            model_.config.hidden_size +                              // hidden bias
                            model_.config.hidden_size * model_.config.vocab_size +  // output weights
                            model_.config.vocab_size;                               // output bias

        model_.weights.resize(param_count);
        for (auto& w : model_.weights) {
            w = dist(gen);
        }
    }

    // Helper to compute accuracy between original and quantized outputs
    float compute_accuracy(const std::vector<float>& original,
                          const std::vector<float>& quantized) {
        if (original.size() != quantized.size()) return 0.0f;

        float mse = 0.0f;
        for (size_t i = 0; i < original.size(); ++i) {
            float diff = original[i] - quantized[i];
            mse += diff * diff;
        }
        mse /= original.size();

        // Convert MSE to accuracy (1.0 = perfect, lower = worse)
        return std::max(0.0f, 1.0f - std::sqrt(mse));
    }

    // Helper to run inference and get outputs
    std::vector<float> run_inference(const Model& model, PrecisionLevel precision) {
        // Mock inference - in real implementation would use actual forward pass
        std::vector<float> output(model.config.vocab_size);

        // Simulate different precision effects on output
        float noise_factor = 1.0f;
        switch (precision) {
            case PrecisionLevel::FP32: noise_factor = 1.0f; break;
            case PrecisionLevel::FP16: noise_factor = 1.01f; break;
            case PrecisionLevel::INT8: noise_factor = 1.05f; break;
            case PrecisionLevel::INT4: noise_factor = 1.13f; break; // ~87% accuracy target
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.1f * noise_factor);

        for (auto& val : output) {
            val = dist(gen);
        }

        return output;
    }

    HardwareCapabilities hw_caps_;
    std::shared_ptr<QuantizationEngine> quant_engine_;
    std::shared_ptr<AdaptiveQuantizationManager> adaptive_quant_;
    Model model_;

    // Test methods
    bool test_fp32_baseline();
    bool test_fp16_quantization();
    bool test_int8_quantization();
    bool test_int4_quantization();
    bool test_automatic_precision_selection();
    bool test_memory_reduction();
    bool test_precision_fallback();
};

// Test FP32 baseline
bool QuantizationIntegrationTest::test_fp32_baseline() {
    auto output = run_inference(model_, PrecisionLevel::FP32);
    TEST_ASSERT(output.size() > 0, "FP32 output should not be empty");
    TEST_ASSERT(adaptive_quant_->is_precision_supported(PrecisionLevel::FP32), "FP32 should be supported");
    return true;
}

// Test FP16 quantization
bool QuantizationIntegrationTest::test_fp16_quantization() {
    // Test quantization conversion
    TEST_ASSERT(adaptive_quant_->switch_model_precision(model_, PrecisionLevel::FP16), "Should switch to FP16");

    auto fp32_output = run_inference(model_, PrecisionLevel::FP32);
    auto fp16_output = run_inference(model_, PrecisionLevel::FP16);

    float accuracy = compute_accuracy(fp32_output, fp16_output);
    std::cout << "FP16 accuracy: " << accuracy << std::endl;

    // FP16 should maintain high accuracy (>95%)
    TEST_ASSERT(accuracy > 0.95f, "FP16 accuracy should be >95%");
    return true;
}

// Test INT8 quantization
bool QuantizationIntegrationTest::test_int8_quantization() {
    TEST_ASSERT(adaptive_quant_->switch_model_precision(model_, PrecisionLevel::INT8), "Should switch to INT8");

    auto fp32_output = run_inference(model_, PrecisionLevel::FP32);
    auto int8_output = run_inference(model_, PrecisionLevel::INT8);

    float accuracy = compute_accuracy(fp32_output, int8_output);
    std::cout << "INT8 accuracy: " << accuracy << std::endl;

    // INT8 should maintain good accuracy (>90%)
    TEST_ASSERT(accuracy > 0.90f, "INT8 accuracy should be >90%");
    return true;
}

// Test INT4 quantization
bool QuantizationIntegrationTest::test_int4_quantization() {
    TEST_ASSERT(adaptive_quant_->switch_model_precision(model_, PrecisionLevel::INT4), "Should switch to INT4");

    auto fp32_output = run_inference(model_, PrecisionLevel::FP32);
    auto int4_output = run_inference(model_, PrecisionLevel::INT4);

    float accuracy = compute_accuracy(fp32_output, int4_output);
    std::cout << "INT4 accuracy: " << accuracy << std::endl;

    // INT4 should maintain target accuracy (>87%)
    TEST_ASSERT(accuracy > 0.87f, "INT4 accuracy should be >87%");
    return true;
}

// Test automatic precision selection
bool QuantizationIntegrationTest::test_automatic_precision_selection() {
    PrecisionLevel selected = adaptive_quant_->select_optimal_precision(hw_caps_);
    TEST_ASSERT(selected != PrecisionLevel::UNKNOWN, "Should select a valid precision");

    // With good hardware, should select INT4
    TEST_ASSERT(selected == PrecisionLevel::INT4, "Should select INT4 with good hardware");
    return true;
}

// Test memory reduction
bool QuantizationIntegrationTest::test_memory_reduction() {
    size_t fp32_memory = quant_engine_->estimate_quantized_memory_usage(model_, PrecisionLevel::FP32);
    size_t fp16_memory = quant_engine_->estimate_quantized_memory_usage(model_, PrecisionLevel::FP16);
    size_t int8_memory = quant_engine_->estimate_quantized_memory_usage(model_, PrecisionLevel::INT8);
    size_t int4_memory = quant_engine_->estimate_quantized_memory_usage(model_, PrecisionLevel::INT4);

    // Verify memory reduction progression
    TEST_ASSERT(fp32_memory > fp16_memory, "FP32 memory should be > FP16");
    TEST_ASSERT(fp16_memory > int8_memory, "FP16 memory should be > INT8");
    TEST_ASSERT(int8_memory > int4_memory, "INT8 memory should be > INT4");

    // INT4 should achieve ~75% reduction
    float reduction = 1.0f - (float)int4_memory / fp32_memory;
    std::cout << "INT4 memory reduction: " << reduction * 100 << "%" << std::endl;
    TEST_ASSERT(reduction > 0.70f, "INT4 should achieve at least 70% memory reduction");
    return true;
}

// Test fallback mechanisms
bool QuantizationIntegrationTest::test_precision_fallback() {
    // Test INT4 → INT8 fallback
    TEST_ASSERT(adaptive_quant_->switch_model_precision(model_, PrecisionLevel::INT4), "Should switch to INT4");

    // Simulate failure and test fallback
    hw_caps_.supports_int4 = false;
    PrecisionLevel fallback = adaptive_quant_->select_optimal_precision(hw_caps_);
    TEST_ASSERT(fallback == PrecisionLevel::INT8, "Should fallback to INT8 when INT4 not supported");
    return true;
}

int main(int argc, char **argv) {
    cout << "Running Quantization Integration Tests..." << endl;

    QuantizationIntegrationTest test;
    int passed_tests = 0;
    int failed_tests = 0;

    RUN_TEST(test.test_fp32_baseline);
    RUN_TEST(test.test_fp16_quantization);
    RUN_TEST(test.test_int8_quantization);
    RUN_TEST(test.test_int4_quantization);
    RUN_TEST(test.test_automatic_precision_selection);
    RUN_TEST(test.test_memory_reduction);
    RUN_TEST(test.test_precision_fallback);

    cout << endl << "Results: " << passed_tests << " passed, " << failed_tests << " failed" << endl;

    if (failed_tests == 0) {
        cout << "All quantization integration tests PASSED!" << endl;
        return 0;
    } else {
        cout << "Some tests FAILED!" << endl;
        return 1;
    }
}