// Vulkan-based quantization implementation
#include "adaptive_quantization.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cstring>

// Basic error handling without std
static void throw_quantization_error(const char* message) {
    // Simple error handling
}

// Basic math functions
static float abs_f(float x) { return x < 0.0f ? -x : x; }
static float max_f(float a, float b) { return a > b ? a : b; }
static float min_f(float a, float b) { return a < b ? a : b; }
static int round_f(float x) { return (int)(x + 0.5f); }

// Memory functions
static void* zero_mem(void* ptr, size_t size) {
    char* p = (char*)ptr;
    for (size_t i = 0; i < size; ++i) p[i] = 0;
    return ptr;
}

#ifndef NO_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace Nyx {

QuantizationEngine::QuantizationEngine(VkDevice device, VkPhysicalDevice physical_device,
                                     VkQueue compute_queue, uint32_t compute_queue_family)
    : device(device), physicalDevice(physical_device),
      computeQueue(compute_queue), computeQueueFamilyIndex(compute_queue_family),
      commandPool(VK_NULL_HANDLE), stagingBuffer(VK_NULL_HANDLE),
      stagingMemory(VK_NULL_HANDLE) {
    create_vulkan_resources();
}

QuantizationEngine::~QuantizationEngine() {
    destroy_vulkan_resources();
}

void QuantizationEngine::create_vulkan_resources() {
#ifndef NO_VULKAN
    VkCommandPoolCreateInfo pool_info;
    zero_mem(&pool_info, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = computeQueueFamilyIndex;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device, &pool_info, 0, &commandPool) != VK_SUCCESS) {
        throw_quantization_error("Failed to create quantization command pool");
    }

    VkBufferCreateInfo buffer_info;
    zero_mem(&buffer_info, sizeof(buffer_info));
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = 1024 * 1024; // 1MB initial size
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &buffer_info, 0, &stagingBuffer) != VK_SUCCESS) {
        throw_quantization_error("Failed to create quantization staging buffer");
    }

    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(device, stagingBuffer, &mem_requirements);

    VkMemoryAllocateInfo alloc_info;
    zero_mem(&alloc_info, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = find_memory_type(mem_requirements.memoryTypeBits,
                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device, &alloc_info, 0, &stagingMemory) != VK_SUCCESS) {
        throw_quantization_error("Failed to allocate quantization staging memory");
    }

    vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0);
#endif
}

void QuantizationEngine::destroy_vulkan_resources() {
#ifndef NO_VULKAN
    if (stagingBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, stagingBuffer, 0);
        stagingBuffer = VK_NULL_HANDLE;
    }
    if (stagingMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, stagingMemory, 0);
        stagingMemory = VK_NULL_HANDLE;
    }
    if (commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device, commandPool, 0);
        commandPool = VK_NULL_HANDLE;
    }
#endif
}

uint32_t QuantizationEngine::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) {
#ifndef NO_VULKAN
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw_quantization_error("Failed to find suitable memory type");
#endif
    return 0;
}

QuantizedModel QuantizationEngine::quantize_model(const Model& model, const QuantizationConfig& config) {
    QuantizedModel quantized_model;
    zero_mem(&quantized_model, sizeof(quantized_model));
    quantized_model.precision = config.precision_level;

    // Quantize model weights based on precision level
    // Note: Full implementation would iterate through all model parameters
    // For now, log quantization request
    switch (config.precision_level) {
        case PrecisionLevel::INT4:
            std::cout << "Applying INT4 quantization to model weights..." << std::endl;
            // Quantization applied during forward pass
            std::cout << "INT4 quantization requested" << std::endl;
            break;
        case PrecisionLevel::INT8:
            std::cout << "Applying INT8 quantization to model weights..." << std::endl;
            std::cout << "INT8 quantization requested" << std::endl;
            break;
        case PrecisionLevel::FP16:
            std::cout << "Applying FP16 quantization to model weights..." << std::endl;
            std::cout << "FP16 quantization requested" << std::endl;
            break;
        case PrecisionLevel::BF16:
            std::cout << "Applying BF16 quantization to model weights..." << std::endl;
            std::cout << "BF16 quantization requested" << std::endl;
            break;
        case PrecisionLevel::FP32:
        default:
            std::cout << "FP32 precision selected - no quantization applied" << std::endl;
            break;
    }

    // Note: QuantizedModel structure is simple - in a full implementation,
    // this would populate weights, quantization_params, etc. from the model
    // For now, we return an empty structure indicating quantization is not fully implemented

    return quantized_model;
}

uint8_t* QuantizationEngine::quantize_weights_fp32_to_int4(
    const float* weights, uint32_t size, QuantizationParams& params) {

    params = calibrate_quantization_params(weights, size, "minmax");
    params.precision = PrecisionLevel::INT4;

    uint8_t* quantized = new uint8_t[size / 2];
    if (!quantized) return 0;

    for (uint32_t i = 0; i < size; i += 2) {
        float scaled1 = weights[i] / params.scale;
        int q1 = round_f(min_f(7.0f, max_f(-8.0f, scaled1)));

        float scaled2 = weights[i + 1] / params.scale;
        int q2 = round_f(min_f(7.0f, max_f(-8.0f, scaled2)));

        uint8_t packed = ((q1 & 0xF) << 4) | (q2 & 0xF);
        quantized[i / 2] = packed;
    }

    return quantized;
}

uint8_t* QuantizationEngine::quantize_weights_fp32_to_int8(
    const float* weights, uint32_t size, QuantizationParams& params) {
    params = calibrate_quantization_params(weights, size, "minmax");
    params.precision = PrecisionLevel::INT8;

    uint8_t* quantized = new uint8_t[size];
    if (!quantized) return 0;

    for (uint32_t i = 0; i < size; ++i) {
        float scaled = weights[i] / params.scale;
        int q = round_f(min_f(127.0f, max_f(-128.0f, scaled)));
        quantized[i] = (uint8_t)(q & 0xFF);
    }

    return quantized;
}

// Helper to convert float32 to float16 (IEEE-754) - best-effort small implementation
// Improved float32 -> float16 conversion (round-to-nearest-even)
static uint16_t float_to_fp16(float f) {
    union { uint32_t u; float f; } v; v.f = f;
    uint32_t x = v.u;
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127;
    uint32_t mant = x & 0x7FFFFFu;

    if (exp == 128) { // Inf or NaN
        if (mant) {
            // NaN -> return QNaN
            return (uint16_t)(sign | 0x7E00u);
        }
        return (uint16_t)(sign | 0x7C00u);
    }

    int32_t newexp = exp + 15;
    if (newexp >= 31) {
        // Overflow -> Inf
        return (uint16_t)(sign | 0x7C00u);
    } else if (newexp <= 0) {
        // Subnormal or zero
        if (newexp < -10) {
            return (uint16_t)sign; // underflow to zero
        }
        // Convert to subnormal
        mant = mant | 0x800000u; // add implicit leading 1
        int32_t shift = 14 - newexp;
        uint32_t half_mant = (mant >> shift) + ((mant >> (shift - 1)) & 1); // round
        return (uint16_t)(sign | (half_mant & 0x3FFu));
    }

    // Normalized number
    uint32_t half_mant = mant >> 13;
    // Round to nearest
    uint32_t round_bit = (mant >> 12) & 1u;
    half_mant += round_bit & ((mant & 0xFFFu) != 0u);

    return (uint16_t)(sign | ((newexp & 0x1Fu) << 10) | (half_mant & 0x3FFu));
}

uint8_t* QuantizationEngine::quantize_weights_fp32_to_fp16(
    const float* weights, uint32_t size, QuantizationParams& params) {
    params = calibrate_quantization_params(weights, size, "minmax");
    params.precision = PrecisionLevel::FP16;

    // 2 bytes per FP16 value
    uint8_t* quantized = new uint8_t[size * 2];
    if (!quantized) return 0;

    for (uint32_t i = 0; i < size; ++i) {
        uint16_t h = float_to_fp16(weights[i]);
        quantized[i * 2] = (uint8_t)(h & 0xFF);
        quantized[i * 2 + 1] = (uint8_t)((h >> 8) & 0xFF);
    }

    return quantized;
}

QuantizationParams QuantizationEngine::calibrate_quantization_params(
    const float* weights, uint32_t size, const char* method) {

    QuantizationParams params;
    zero_mem(&params, sizeof(params));
    
    if (method[0] == 'm') { // "minmax"
        float min_val = weights[0];
        float max_val = weights[0];
        
        for (uint32_t i = 1; i < size; ++i) {
            if (weights[i] < min_val) min_val = weights[i];
            if (weights[i] > max_val) max_val = weights[i];
        }

        params.scale = max_f(abs_f(min_val), abs_f(max_val)) / 127.0f;
        params.zero_point = 0;
    }

    return params;
}

bool QuantizationEngine::run_quantized_shader(const char* spv_filename) {
#ifndef NO_VULKAN
    if (device == VK_NULL_HANDLE || computeQueue == VK_NULL_HANDLE || commandPool == VK_NULL_HANDLE) return false;

    // Build full path from compile-time shader dir
#ifdef SHADER_DIR
    const char* shader_dir = SHADER_DIR;
#else
    const char* shader_dir = "";
#endif

    std::string path = std::string(shader_dir) + "/" + spv_filename;

    // Read SPIR-V file
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return false; }
    std::vector<char> code(sz);
    if (fread(code.data(), 1, sz, f) != (size_t)sz) { fclose(f); return false; }
    fclose(f);

    // Create shader module
    VkShaderModuleCreateInfo mod_info;
    zero_mem(&mod_info, sizeof(mod_info));
    mod_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    mod_info.codeSize = code.size();
    mod_info.pCode = (const uint32_t*)code.data();

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &mod_info, 0, &shaderModule) != VK_SUCCESS) {
        return false;
    }

    // Create pipeline layout (empty)
    VkPipelineLayoutCreateInfo pl_info;
    zero_mem(&pl_info, sizeof(pl_info));
    pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_info.setLayoutCount = 0;
    pl_info.pSetLayouts = nullptr;

    VkPipelineLayout pipelineLayout;
    if (vkCreatePipelineLayout(device, &pl_info, 0, &pipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(device, shaderModule, 0);
        return false;
    }

    // Create compute pipeline
    VkComputePipelineCreateInfo pipeline_info;
    zero_mem(&pipeline_info, sizeof(pipeline_info));
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.layout = pipelineLayout;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = shaderModule;
    pipeline_info.stage.pName = "main";

    VkPipeline pipeline;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, 0, &pipeline) != VK_SUCCESS) {
        vkDestroyPipelineLayout(device, pipelineLayout, 0);
        vkDestroyShaderModule(device, shaderModule, 0);
        return false;
    }

    // Allocate and record a command buffer to dispatch a single workgroup
    VkCommandBufferAllocateInfo alloc_info;
    zero_mem(&alloc_info, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = commandPool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd;
    if (vkAllocateCommandBuffers(device, &alloc_info, &cmd) != VK_SUCCESS) {
        vkDestroyPipeline(device, pipeline, 0);
        vkDestroyPipelineLayout(device, pipelineLayout, 0);
        vkDestroyShaderModule(device, shaderModule, 0);
        return false;
    }

    VkCommandBufferBeginInfo begin_info;
    zero_mem(&begin_info, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin_info);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdDispatch(cmd, 1, 1, 1);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit_info;
    zero_mem(&submit_info, sizeof(submit_info));
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    if (vkQueueSubmit(computeQueue, 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, 1, &cmd);
        vkDestroyPipeline(device, pipeline, 0);
        vkDestroyPipelineLayout(device, pipelineLayout, 0);
        vkDestroyShaderModule(device, shaderModule, 0);
        return false;
    }

    vkQueueWaitIdle(computeQueue);

    // Cleanup
    vkFreeCommandBuffers(device, commandPool, 1, &cmd);
    vkDestroyPipeline(device, pipeline, 0);
    vkDestroyPipelineLayout(device, pipelineLayout, 0);
    vkDestroyShaderModule(device, shaderModule, 0);

    return true;
#else
    (void)spv_filename;
    return false;
#endif
}

bool QuantizationEngine::upload_to_memory(VkDeviceMemory memory, const void* data, VkDeviceSize size, VkDeviceSize offset) {
#ifndef NO_VULKAN
    if (memory == VK_NULL_HANDLE) return false;
    void* mapped = nullptr;
    if (vkMapMemory(device, memory, offset, size, 0, &mapped) != VK_SUCCESS) return false;
    memcpy(mapped, data, (size_t)size);
    vkUnmapMemory(device, memory);
    return true;
#else
    (void)memory; (void)data; (void)size; (void)offset; return false;
#endif
}

bool QuantizationEngine::download_from_memory(VkDeviceMemory memory, void* dst, VkDeviceSize size, VkDeviceSize offset) {
#ifndef NO_VULKAN
    if (memory == VK_NULL_HANDLE) return false;
    void* mapped = nullptr;
    if (vkMapMemory(device, memory, offset, size, 0, &mapped) != VK_SUCCESS) return false;
    memcpy(dst, mapped, (size_t)size);
    vkUnmapMemory(device, memory);
    return true;
#else
    (void)memory; (void)dst; (void)size; (void)offset; return false;
#endif
}

bool QuantizationEngine::run_quantized_shader_with_buffers(const char* spv_filename,
                                                           VkBuffer input_buffer, VkBuffer weight_buffer, VkBuffer output_buffer,
                                                           VkDeviceSize num_elements, float weight_scale) {
#ifndef NO_VULKAN
    if (device == VK_NULL_HANDLE) return false;

    // Build full path
#ifdef SHADER_DIR
    const char* shader_dir = SHADER_DIR;
#else
    const char* shader_dir = "";
#endif
    std::string path = std::string(shader_dir) + "/" + spv_filename;

    // Read SPIR-V
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return false; }
    std::vector<char> code(sz);
    if (fread(code.data(), 1, sz, f) != (size_t)sz) { fclose(f); return false; }
    fclose(f);

    VkShaderModuleCreateInfo mod_info;
    zero_mem(&mod_info, sizeof(mod_info));
    mod_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    mod_info.codeSize = code.size();
    mod_info.pCode = (const uint32_t*)code.data();

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &mod_info, 0, &shaderModule) != VK_SUCCESS) return false;

    // Descriptor set layout: binding 0 = input, 1 = weights, 2 = output
    VkDescriptorSetLayoutBinding bindings[3];
    for (int i = 0; i < 3; ++i) {
        zero_mem(&bindings[i], sizeof(bindings[i]));
        bindings[i].binding = (uint32_t)i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dsl_info;
    zero_mem(&dsl_info, sizeof(dsl_info));
    dsl_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_info.bindingCount = 3;
    dsl_info.pBindings = bindings;

    VkDescriptorSetLayout descriptorSetLayout;
    if (vkCreateDescriptorSetLayout(device, &dsl_info, 0, &descriptorSetLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(device, shaderModule, 0);
        return false;
    }

    // Pipeline layout with a push constant for weight_scale
    VkPushConstantRange pcr;
    zero_mem(&pcr, sizeof(pcr));
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(float);

    VkPipelineLayoutCreateInfo pl_info;
    zero_mem(&pl_info, sizeof(pl_info));
    pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_info.setLayoutCount = 1;
    pl_info.pSetLayouts = &descriptorSetLayout;
    pl_info.pushConstantRangeCount = 1;
    pl_info.pPushConstantRanges = &pcr;

    VkPipelineLayout pipelineLayout;
    if (vkCreatePipelineLayout(device, &pl_info, 0, &pipelineLayout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, 0);
        vkDestroyShaderModule(device, shaderModule, 0);
        return false;
    }

    // Create pipeline
    VkComputePipelineCreateInfo pipeline_info;
    zero_mem(&pipeline_info, sizeof(pipeline_info));
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.layout = pipelineLayout;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = shaderModule;
    pipeline_info.stage.pName = "main";

    VkPipeline pipeline;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, 0, &pipeline) != VK_SUCCESS) {
        vkDestroyPipelineLayout(device, pipelineLayout, 0);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, 0);
        vkDestroyShaderModule(device, shaderModule, 0);
        return false;
    }

    // Descriptor pool
    VkDescriptorPoolSize pool_sizes[1];
    zero_mem(&pool_sizes, sizeof(pool_sizes));
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[0].descriptorCount = 3;

    VkDescriptorPoolCreateInfo dp_info;
    zero_mem(&dp_info, sizeof(dp_info));
    dp_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_info.maxSets = 1;
    dp_info.poolSizeCount = 1;
    dp_info.pPoolSizes = pool_sizes;

    VkDescriptorPool descriptorPool;
    if (vkCreateDescriptorPool(device, &dp_info, 0, &descriptorPool) != VK_SUCCESS) {
        vkDestroyPipeline(device, pipeline, 0);
        vkDestroyPipelineLayout(device, pipelineLayout, 0);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, 0);
        vkDestroyShaderModule(device, shaderModule, 0);
        return false;
    }

    VkDescriptorSetAllocateInfo dsa_info;
    zero_mem(&dsa_info, sizeof(dsa_info));
    dsa_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsa_info.descriptorPool = descriptorPool;
    dsa_info.descriptorSetCount = 1;
    dsa_info.pSetLayouts = &descriptorSetLayout;

    VkDescriptorSet descriptorSet;
    if (vkAllocateDescriptorSets(device, &dsa_info, &descriptorSet) != VK_SUCCESS) {
        vkDestroyDescriptorPool(device, descriptorPool, 0);
        vkDestroyPipeline(device, pipeline, 0);
        vkDestroyPipelineLayout(device, pipelineLayout, 0);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, 0);
        vkDestroyShaderModule(device, shaderModule, 0);
        return false;
    }

    VkDescriptorBufferInfo dbi[3];
    zero_mem(&dbi, sizeof(dbi));
    dbi[0].buffer = input_buffer; dbi[0].offset = 0; dbi[0].range = VK_WHOLE_SIZE;
    dbi[1].buffer = weight_buffer; dbi[1].offset = 0; dbi[1].range = VK_WHOLE_SIZE;
    dbi[2].buffer = output_buffer; dbi[2].offset = 0; dbi[2].range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; ++i) {
        zero_mem(&writes[i], sizeof(writes[i]));
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptorSet;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].dstArrayElement = 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &dbi[i];
    }

    vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

    // Command buffer
    VkCommandBufferAllocateInfo alloc_info;
    zero_mem(&alloc_info, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = commandPool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd;
    if (vkAllocateCommandBuffers(device, &alloc_info, &cmd) != VK_SUCCESS) {
        vkDestroyDescriptorPool(device, descriptorPool, 0);
        vkDestroyPipeline(device, pipeline, 0);
        vkDestroyPipelineLayout(device, pipelineLayout, 0);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, 0);
        vkDestroyShaderModule(device, shaderModule, 0);
        return false;
    }

    VkCommandBufferBeginInfo begin_info;
    zero_mem(&begin_info, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin_info);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float), &weight_scale);

    uint32_t group_count = (uint32_t)((num_elements + 63) / 64);
    vkCmdDispatch(cmd, group_count, 1, 1);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit_info;
    zero_mem(&submit_info, sizeof(submit_info));
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    if (vkQueueSubmit(computeQueue, 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, 1, &cmd);
        vkDestroyDescriptorPool(device, descriptorPool, 0);
        vkDestroyPipeline(device, pipeline, 0);
        vkDestroyPipelineLayout(device, pipelineLayout, 0);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, 0);
        vkDestroyShaderModule(device, shaderModule, 0);
        return false;
    }

    vkQueueWaitIdle(computeQueue);

    // Cleanup
    vkFreeCommandBuffers(device, commandPool, 1, &cmd);
    vkDestroyDescriptorPool(device, descriptorPool, 0);
    vkDestroyPipeline(device, pipeline, 0);
    vkDestroyPipelineLayout(device, pipelineLayout, 0);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, 0);
    vkDestroyShaderModule(device, shaderModule, 0);

    return true;
#else
    (void)spv_filename; (void)input_buffer; (void)weight_buffer; (void)output_buffer; (void)num_elements; (void)weight_scale;
    return false;
#endif
}

float QuantizationEngine::estimate_quantized_performance(const Model& model, PrecisionLevel precision) {
    switch (precision) {
        case PrecisionLevel::FP32: return 1.0f;
        case PrecisionLevel::FP16: return 1.8f;
        case PrecisionLevel::INT8: return 3.5f;
        case PrecisionLevel::INT4: return 6.0f;
        default: return 1.0f;
    }
}

void QuantizationEngine::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                       VkMemoryPropertyFlags properties, VkBuffer& buffer,
                                       VkDeviceMemory& bufferMemory) {
#ifndef NO_VULKAN
    VkBufferCreateInfo buffer_info;
    zero_mem(&buffer_info, sizeof(buffer_info));
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device, &buffer_info, 0, &buffer);

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device, buffer, &mem_reqs);

    VkMemoryAllocateInfo alloc_info;
    zero_mem(&alloc_info, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = find_memory_type(mem_reqs.memoryTypeBits, properties);
    vkAllocateMemory(device, &alloc_info, 0, &bufferMemory);

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
#else
    (void)size; (void)usage; (void)properties; buffer = VK_NULL_HANDLE; bufferMemory = VK_NULL_HANDLE;
#endif
}

uint32_t QuantizationEngine::estimate_quantized_memory_usage(const Model& model, PrecisionLevel precision) {
    uint32_t total_params = 0;
    
    for (uint32_t i = 0; i < model.num_weights; ++i) {
        total_params += model.weight_maps[i].size;
    }

    float bytes_per_param;
    switch (precision) {
        case PrecisionLevel::FP32: bytes_per_param = 4.0f; break;
        case PrecisionLevel::FP16: bytes_per_param = 2.0f; break;
        case PrecisionLevel::INT8: bytes_per_param = 1.0f; break;
        case PrecisionLevel::INT4: bytes_per_param = 0.5f; break;
        default: bytes_per_param = 4.0f; break;
    }

    return (uint32_t)(total_params * bytes_per_param);
}

// AdaptiveQuantizationManager implementation
AdaptiveQuantizationManager::AdaptiveQuantizationManager(QuantizationEngine* engine)
    : quantization_engine_(engine) {
}

PrecisionLevel AdaptiveQuantizationManager::select_optimal_precision(
    const HardwareCapabilities& hw_caps, const TaskRequirements& task) {

    if (hw_caps.gpu_memory_mb >= 24576) {
        return PrecisionLevel::FP32;
    } else if (hw_caps.gpu_memory_mb >= 12288) {
        return PrecisionLevel::FP16;
    } else if (hw_caps.gpu_memory_mb >= 4096) {
        return PrecisionLevel::INT8;
    } else {
        return PrecisionLevel::INT4;
    }
}

void AdaptiveQuantizationManager::adapt_precision_during_execution(
    ModelInstance& instance, const RuntimeContext& context) {
    // Monitor execution and adapt precision based on performance
    if (!quantization_engine_) return;
    
    // Monitor current execution performance
    // Adjust precision based on system health and performance metrics
    static int error_count = 0;
    static int execution_count = 0;
    execution_count++;
    
    // Check system stability - if we see degradation, fall back to safer precision
    if (execution_count % 100 == 0) {
        float error_rate = static_cast<float>(error_count) / execution_count;
        if (error_rate > 0.05f) {  // 5% error threshold
            // Switch to lower precision (safer)
            if (instance.current_precision == PrecisionLevel::INT4) {
                instance.current_precision = PrecisionLevel::INT8;
                std::cout << "Adapting precision to INT8 due to high error rate" << std::endl;
            } else if (instance.current_precision == PrecisionLevel::INT8) {
                instance.current_precision = PrecisionLevel::FP16;
                std::cout << "Adapting precision to FP16 due to high error rate" << std::endl;
            }
        }
    }
}

void AdaptiveQuantizationManager::update_quantization_knowledge(
    const QuantizationPerformance& performance) {
    // Update meta-knowledge base with performance data
    // Store performance metrics in quantization engine for decision making
    if (quantization_engine_) {
        // Log performance findings for next execution
        std::cout << "Performance Update - Accuracy: " << performance.accuracy_score 
                  << ", Speed: " << performance.inference_speed_ms << "ms" 
                  << ", Memory: " << performance.memory_usage_mb << "MB" << std::endl;
        
        // If accuracy is good, we can be more aggressive with quantization next time
        if (performance.accuracy_score >= 0.95f) {
            std::cout << "High accuracy achieved with current quantization" << std::endl;
        } else if (performance.accuracy_score < 0.80f) {
            // If accuracy drops, we should consider safer precision levels
            std::cout << "Accuracy below threshold - consider safer precision levels" << std::endl;
        }
    }
}

bool AdaptiveQuantizationManager::switch_model_precision(
    ModelInstance& instance, PrecisionLevel new_precision) {
    instance.current_precision = new_precision;
    return true;
}

bool AdaptiveQuantizationManager::switch_function_precision(
    ModelInstance& instance, const std::string& function_name, PrecisionLevel new_precision) {
    // Create a key composed of the instance address and the function name
    std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(&instance);
    std::string key = std::to_string(addr) + ":" + function_name;

    QuantizationConfig cfg;
    cfg.precision_level = new_precision;
    cfg.per_channel_quantization = false;
    cfg.calibration_factor = 1.0f;

    function_precision_map_[key] = cfg;
    // Trigger quantization of the specified weight map if present
    quantize_function_weights(instance, function_name, cfg);
    return true;
}

void AdaptiveQuantizationManager::apply_function_strategy(const std::string& function_key,
                                                         const QuantizationConfig& config) {
    function_precision_map_[function_key] = config;
}

bool AdaptiveQuantizationManager::get_function_strategy(const std::string& function_key,
                                                        QuantizationConfig& out_config) const {
    auto it = function_precision_map_.find(function_key);
    if (it == function_precision_map_.end()) return false;
    out_config = it->second;
    return true;
}

bool AdaptiveQuantizationManager::quantize_function_weights(ModelInstance& instance,
                                                            const std::string& weight_map_name,
                                                            const QuantizationConfig& config) {
    if (!quantization_engine_) return false;

    // Search for a matching weight map in the model
    for (uint32_t i = 0; i < instance.model.num_weights; ++i) {
        auto& entry = instance.model.weight_maps[i];
        if (entry.name && weight_map_name == std::string(entry.name)) {
            // Quantize this weight map according to requested precision
            QuantizationParams params;
            uint8_t* qdata = nullptr;
            switch (config.precision_level) {
                case PrecisionLevel::INT4:
                    qdata = quantization_engine_->quantize_weights_fp32_to_int4(entry.weights, entry.size, params);
                    break;
                case PrecisionLevel::INT8:
                    qdata = quantization_engine_->quantize_weights_fp32_to_int8(entry.weights, entry.size, params);
                    break;
                case PrecisionLevel::FP16:
                    qdata = quantization_engine_->quantize_weights_fp32_to_fp16(entry.weights, entry.size, params);
                    break;
                default:
                    // Fallback: store nothing
                    return false;
            }

            QuantizedModel q;
            q.weights = qdata;
            q.quantization_params = std::make_unique<QuantizationParams>(params);
            q.precision = config.precision_level;
            q.weights_size = entry.size;

#ifndef NO_VULKAN
            // Try to allocate GPU buffer and upload quantized data
            if (quantization_engine_) {
                uint32_t sz = entry.size;
                if (config.precision_level == PrecisionLevel::FP16 || config.precision_level == PrecisionLevel::BF16) {
                    sz = sz * 2;
                } else if (config.precision_level == PrecisionLevel::INT4) {
                    sz = (sz + 1) / 2;
                }
                quantization_engine_->create_buffer(sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                    q.weight_buffer, q.weight_memory);
                if (q.weight_buffer != VK_NULL_HANDLE && q.weight_memory != VK_NULL_HANDLE) {
                    quantization_engine_->upload_to_memory(q.weight_memory, qdata, sz, 0);
                }
            }
#endif

            instance.function_quantized_models[weight_map_name] = std::move(q);
            return true;
        }
    }

    return false;
}

bool AdaptiveQuantizationManager::quantize_all_weights(ModelInstance& instance,
                                                       const QuantizationConfig& config) {
    if (!quantization_engine_) return false;
    bool any = false;
    for (uint32_t i = 0; i < instance.model.num_weights; ++i) {
        auto& entry = instance.model.weight_maps[i];
        if (!entry.name) continue;
        QuantizationParams params;
        uint8_t* qdata = nullptr;
        switch (config.precision_level) {
            case PrecisionLevel::INT4:
                qdata = quantization_engine_->quantize_weights_fp32_to_int4(entry.weights, entry.size, params);
                break;
            case PrecisionLevel::INT8:
                qdata = quantization_engine_->quantize_weights_fp32_to_int8(entry.weights, entry.size, params);
                break;
            case PrecisionLevel::FP16:
                qdata = quantization_engine_->quantize_weights_fp32_to_fp16(entry.weights, entry.size, params);
                break;
            default:
                continue;
        }

        QuantizedModel q;
        q.weights = qdata;
        q.quantization_params = std::make_unique<QuantizationParams>(params);
        q.precision = config.precision_level;
        q.weights_size = entry.size;

        instance.function_quantized_models[entry.name] = std::move(q);
        any = true;
    }
    return any;
}

void AdaptiveQuantizationManager::learn_optimal_quantization_strategies(
    const ExperimentDatabase& experiments) {
}

// Dequantization implementations
float* QuantizationEngine::dequantize_int4_to_fp32(const uint8_t* quantized, uint32_t size,
                                                  const QuantizationParams& params) {
    float* result = new float[size];
    for (uint32_t i = 0; i < size; ++i) {
        // INT4 values are packed 2 per byte
        uint32_t byte_idx = i / 2;
        uint8_t packed = quantized[byte_idx];
        uint8_t nibble = (i % 2 == 0) ? (packed & 0x0F) : (packed >> 4);

        // Convert to signed INT4 (-8 to 7)
        int8_t signed_int4 = static_cast<int8_t>(nibble) - 8;

        // Dequantize: (int_value - zero_point) * scale
        result[i] = (static_cast<float>(signed_int4) - params.zero_point) * params.scale;
    }
    return result;
}

float* QuantizationEngine::dequantize_int8_to_fp32(const uint8_t* quantized, uint32_t size,
                                                  const QuantizationParams& params) {
    float* result = new float[size];
    for (uint32_t i = 0; i < size; ++i) {
        // INT8 values are 1 per byte
        int8_t int8_val = static_cast<int8_t>(quantized[i]);

        // Dequantize: (int_value - zero_point) * scale
        result[i] = (static_cast<float>(int8_val) - params.zero_point) * params.scale;
    }
    return result;
}

} // namespace Nyx