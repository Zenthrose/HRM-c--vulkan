#pragma once

#ifndef NO_VULKAN
#include <vulkan/vulkan.h>
#endif

#include "quantization_types.hpp"
#include <unordered_map>
#include <string>

// Forward declarations for complex types to avoid circular dependencies
struct Tensor;
struct Model;

namespace Nyx {

// Vulkan-based Quantization Engine
class QuantizationEngine {
public:
    QuantizationEngine(VkDevice device, VkPhysicalDevice physical_device,
                      VkQueue compute_queue, uint32_t compute_queue_family);
    ~QuantizationEngine();

    // Core quantization methods
    QuantizedModel quantize_model(const Model& model, const QuantizationConfig& config);
    void create_vulkan_resources();
    void destroy_vulkan_resources();

    // Quantization methods with Vulkan acceleration
    uint8_t* quantize_weights_fp32_to_int4(const float* weights, uint32_t size,
                                           QuantizationParams& params);
    uint8_t* quantize_weights_fp32_to_int8(const float* weights, uint32_t size,
                                           QuantizationParams& params);
    uint8_t* quantize_weights_fp32_to_fp16(const float* weights, uint32_t size,
                                           QuantizationParams& params);

    // Dequantization methods
    float* dequantize_int4_to_fp32(const uint8_t* quantized, uint32_t size,
                                   const QuantizationParams& params);
    float* dequantize_int8_to_fp32(const uint8_t* quantized, uint32_t size,
                                   const QuantizationParams& params);

    // Calibration methods
    QuantizationParams calibrate_quantization_params(const float* weights, uint32_t size,
                                                const char* method);

    // Performance estimation
    float estimate_quantized_performance(const Model& model, PrecisionLevel precision);
    uint32_t estimate_quantized_memory_usage(const Model& model, PrecisionLevel precision);

public:
    // Buffer management - exposed for GPU-backed quantization
    void create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties, VkBuffer& buffer,
                      VkDeviceMemory& bufferMemory);

    // Upload/download helpers for mapped memory
    bool upload_to_memory(VkDeviceMemory memory, const void* data, VkDeviceSize size, VkDeviceSize offset = 0);
    bool download_from_memory(VkDeviceMemory memory, void* dst, VkDeviceSize size, VkDeviceSize offset = 0);

    // Run a shader bound to provided storage buffers: input, weights, output
    bool run_quantized_shader_with_buffers(const char* spv_filename,
                                           VkBuffer input_buffer, VkBuffer weight_buffer, VkBuffer output_buffer,
                                           VkDeviceSize num_elements, float weight_scale);

    // Dispatch a compiled SPIR-V shader by filename (relative to `SHADER_DIR`), minimal compute dispatch
    bool run_quantized_shader(const char* spv_filename);

private:
    // Vulkan resources - match .cpp file member names
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue computeQueue;
    uint32_t computeQueueFamilyIndex;
    VkCommandPool commandPool;
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;
    VkDescriptorPool descriptorPool;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline quantizationPipeline;
    VkCommandBuffer commandBuffer;

    // Helper methods
    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);
  };

// Adaptive quantization manager for hardware-aware optimization
class AdaptiveQuantizationManager {
public:
    AdaptiveQuantizationManager(QuantizationEngine* engine);

    // Hardware-aware precision selection
    PrecisionLevel select_optimal_precision(const HardwareCapabilities& hw_caps,
                                          const TaskRequirements& task);

    // Runtime precision adaptation
    void adapt_precision_during_execution(ModelInstance& instance,
                                        const RuntimeContext& context);

    // Performance learning
    void update_quantization_knowledge(const QuantizationPerformance& performance);

    // Model precision switching
    bool switch_model_precision(ModelInstance& instance, PrecisionLevel new_precision);

    // Note: no no-arg shim — prefer explicit model instance + precision

    // Strategy learning
    void learn_optimal_quantization_strategies(const ExperimentDatabase& experiments);

    // Per-function precision control: set/get strategies for individual functions
    bool switch_function_precision(ModelInstance& instance, const std::string& function_name, PrecisionLevel new_precision);
    void apply_function_strategy(const std::string& function_key, const QuantizationConfig& config);
    bool get_function_strategy(const std::string& function_key, QuantizationConfig& out_config) const;
    // Quantize specific function/weight-map or all weight-maps of a model instance
    bool quantize_function_weights(ModelInstance& instance, const std::string& weight_map_name, const QuantizationConfig& config);
    bool quantize_all_weights(ModelInstance& instance, const QuantizationConfig& config);

private:
    QuantizationEngine* quantization_engine_;
    // Per-function strategies keyed by "<instance_ptr>:<function_name>" or global identifiers
    std::unordered_map<std::string, QuantizationConfig> function_precision_map_;
    
    // Performance tracking structures would go here
    // Using minimal implementation for now
};

} // namespace Nyx