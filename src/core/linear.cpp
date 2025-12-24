#include <iostream>
#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include "linear.hpp"
LinearVulkan::LinearVulkan(const LinearConfig& config, VkPhysicalDevice physicalDevice, VkDevice device, VkQueue computeQueue, uint32_t computeQueueFamilyIndex, VkCommandPool commandPool)
    : config(config), physicalDevice(physicalDevice), device(device), computeQueue(computeQueue), computeQueueFamilyIndex(computeQueueFamilyIndex), commandPool(commandPool),
      pipeline(VK_NULL_HANDLE), pipelineLayout(VK_NULL_HANDLE), descriptorSetLayout(VK_NULL_HANDLE), descriptorPool(VK_NULL_HANDLE),
      inputBuffer(VK_NULL_HANDLE), weightBuffer(VK_NULL_HANDLE), biasBuffer(VK_NULL_HANDLE), outputBuffer(VK_NULL_HANDLE), uniformBuffer(VK_NULL_HANDLE),
      inputBufferMemory(VK_NULL_HANDLE), weightBufferMemory(VK_NULL_HANDLE), biasBufferMemory(VK_NULL_HANDLE), outputBufferMemory(VK_NULL_HANDLE), uniformBufferMemory(VK_NULL_HANDLE) {
    std::cout << "Initializing LinearVulkan layer..." << std::endl;
    std::cout << "  Device: " << (device != VK_NULL_HANDLE ? "valid" : "null") << std::endl;
    std::cout << "  Physical device: " << (physicalDevice != VK_NULL_HANDLE ? "valid" : "null") << std::endl;
    if (device != VK_NULL_HANDLE) {
        try {
            init_vulkan_objects();
            std::cout << "  Vulkan objects initialized successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "  ERROR: Failed to initialize Vulkan objects: " << e.what() << std::endl;
            throw;
        }
    } else {
        std::cout << "  Skipping Vulkan initialization (CPU fallback)" << std::endl;
    }
}

LinearVulkan::~LinearVulkan() {
    std::cout << "Destroying LinearVulkan layer..." << std::endl;
    
    // Check if device is still valid before cleanup
    if (device == VK_NULL_HANDLE) {
        return; // Device already destroyed, skip cleanup
    }
    
    // Clean up staging buffers first
    for (auto& staging : stagingBuffers) {
        if (staging.first != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, staging.first, nullptr);
        }
        if (staging.second != VK_NULL_HANDLE) {
            vkFreeMemory(device, staging.second, nullptr);
        }
    }
    stagingBuffers.clear();

    // Clean up persistent buffers
    if (inputBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, inputBuffer, nullptr);
        vkFreeMemory(device, inputBufferMemory, nullptr);
    }
    if (weightBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, weightBuffer, nullptr);
        vkFreeMemory(device, weightBufferMemory, nullptr);
    }
    if (biasBuffer != VK_NULL_HANDLE && config.bias) {
        vkDestroyBuffer(device, biasBuffer, nullptr);
        vkFreeMemory(device, biasBufferMemory, nullptr);
    }
    if (outputBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, outputBuffer, nullptr);
        vkFreeMemory(device, outputBufferMemory, nullptr);
    }
    if (uniformBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, uniformBuffer, nullptr);
        vkFreeMemory(device, uniformBufferMemory, nullptr);
    }

    if (descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    }
    if (pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, pipeline, nullptr);
    }
    if (pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    }
    if (descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    }
}

void LinearVulkan::init_vulkan_objects() {
    // Create persistent buffers
    VkBufferUsageFlags storage_buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags memory_properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // We'll allocate buffers dynamically in forward() based on input size
    // For now, create uniform buffer
    createBuffer(sizeof(LinearParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 uniformBuffer, uniformBufferMemory);

    // Descriptor pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}, // input, weights, bias?, output
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}
    };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool");
    }

    // Descriptor set layout
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // input
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // weights
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // bias (optional)
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // output
        {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}  // params
    };

    VkDescriptorSetLayoutCreateInfo layoutCI{};
    layoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCI.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutCI.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutCI, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set");
    }

    createComputePipeline();
}

void LinearVulkan::createComputePipeline() {
    // Load shader
    std::ifstream file("shaders/linear.spv", std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: shaders/linear.spv");
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    VkShaderModuleCreateInfo shaderModuleCI{};
    shaderModuleCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCI.codeSize = buffer.size();
    shaderModuleCI.pCode = reinterpret_cast<const uint32_t*>(buffer.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &shaderModuleCI, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }

    // Pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutCI{};
    pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCI.setLayoutCount = 1;
    pipelineLayoutCI.pSetLayouts = &descriptorSetLayout;

    if (vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(device, shaderModule, nullptr);
        throw std::runtime_error("Failed to create pipeline layout");
    }

    // Pipeline
    VkComputePipelineCreateInfo pipelineCI{};
    pipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCI.layout = pipelineLayout;
    pipelineCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineCI.stage.module = shaderModule;
    pipelineCI.stage.pName = "main";

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, shaderModule, nullptr);
        throw std::runtime_error("Failed to create compute pipeline");
    }

    vkDestroyShaderModule(device, shaderModule, nullptr);
}

void LinearVulkan::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        vkDestroyBuffer(device, buffer, nullptr);
        throw std::runtime_error("Failed to allocate buffer memory");
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

void LinearVulkan::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    // Check if device is still valid before operations
    if (device == VK_NULL_HANDLE || computeQueue == VK_NULL_HANDLE) {
        return; // Device already destroyed, skip copy
    }
    
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    if (vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        throw std::runtime_error("failed to submit linear copy command buffer!");
    }
    if (vkQueueWaitIdle(computeQueue) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        throw std::runtime_error("failed to wait for linear copy queue idle!");
    }

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

uint32_t LinearVulkan::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type");
}

Tensor LinearVulkan::forward(const Tensor& input) {

    if (device == VK_NULL_HANDLE) {
        // Fallback to CPU implementation if no Vulkan device
        size_t batch_size = input.data.size() / config.in_features;
        Tensor output;
        output.shape = {static_cast<unsigned int>(batch_size), config.out_features};
        output.data.resize(batch_size * config.out_features);

        // Simple identity weights for demo
        std::vector<float> weights(config.out_features * config.in_features, 0.0f);
        for (size_t i = 0; i < std::min(static_cast<size_t>(config.in_features), static_cast<size_t>(config.out_features)); ++i) {
            weights[i * config.in_features + i] = 1.0f;
        }

        // Matrix multiplication: output = input @ weights.T
        for (size_t batch = 0; batch < batch_size; ++batch) {
            for (size_t out = 0; out < static_cast<size_t>(config.out_features); ++out) {
                float sum = 0.0f;
                for (size_t in = 0; in < static_cast<size_t>(config.in_features); ++in) {
                    sum += input.data[batch * config.in_features + in] * weights[out * config.in_features + in];
                }
                output.data[batch * config.out_features + out] = sum;
            }
    }

    return output;
}



    // Vulkan implementation
    size_t batch_size = input.data.size() / config.in_features;
    size_t total_elements = batch_size * config.out_features;

    Tensor output;
    output.shape = {static_cast<unsigned int>(batch_size), config.out_features};
    output.data.resize(total_elements);

    // Calculate buffer sizes
    VkDeviceSize input_size = input.data.size() * sizeof(float);
    VkDeviceSize weight_size = config.out_features * config.in_features * sizeof(float);
    VkDeviceSize bias_size = config.bias ? config.out_features * sizeof(float) : 0;
    VkDeviceSize output_size = output.data.size() * sizeof(float);

    // Create buffers
    VkBuffer inputStaging, weightStaging, biasStaging;
    VkDeviceMemory inputStagingMemory, weightStagingMemory, biasStagingMemory;

    createBuffer(input_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 inputStaging, inputStagingMemory);
    stagingBuffers.push_back({inputStaging, inputStagingMemory});

    createBuffer(weight_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 weightStaging, weightStagingMemory);
    stagingBuffers.push_back({weightStaging, weightStagingMemory});

    if (config.bias) {
        createBuffer(bias_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     biasStaging, biasStagingMemory);
        stagingBuffers.push_back({biasStaging, biasStagingMemory});
    }

    // Create device buffers
    createBuffer(input_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, inputBuffer, inputBufferMemory);
    createBuffer(weight_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, weightBuffer, weightBufferMemory);
    createBuffer(output_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, outputBuffer, outputBufferMemory);

    if (config.bias) {
        createBuffer(bias_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, biasBuffer, biasBufferMemory);
    }

    // Initialize weights (identity for demo)
    std::vector<float> weights(config.out_features * config.in_features, 0.0f);
    for (size_t i = 0; i < std::min(static_cast<size_t>(config.in_features), static_cast<size_t>(config.out_features)); ++i) {
        weights[i * config.in_features + i] = 1.0f;
    }

    // Initialize bias (zeros)
    std::vector<float> biases(config.out_features, 0.0f);

    // Copy data to staging buffers
    void* inputData;
    vkMapMemory(device, inputStagingMemory, 0, input_size, 0, &inputData);
    memcpy(inputData, input.data.data(), input_size);
    vkUnmapMemory(device, inputStagingMemory);

    void* weightData;
    vkMapMemory(device, weightStagingMemory, 0, weight_size, 0, &weightData);
    memcpy(weightData, weights.data(), weight_size);
    vkUnmapMemory(device, weightStagingMemory);

    if (config.bias) {
        void* biasData;
        vkMapMemory(device, biasStagingMemory, 0, bias_size, 0, &biasData);
        memcpy(biasData, biases.data(), bias_size);
        vkUnmapMemory(device, biasStagingMemory);
    }

    // Copy to device buffers
    copyBuffer(inputStaging, inputBuffer, input_size);
    copyBuffer(weightStaging, weightBuffer, weight_size);
    if (config.bias) {
        copyBuffer(biasStaging, biasBuffer, bias_size);
    }

    // Update uniform buffer
    LinearParams params = {static_cast<uint32_t>(batch_size), config.in_features, config.out_features, config.bias ? 1u : 0u};
    void* uniformData;
    vkMapMemory(device, uniformBufferMemory, 0, sizeof(LinearParams), 0, &uniformData);
    memcpy(uniformData, &params, sizeof(LinearParams));
    vkUnmapMemory(device, uniformBufferMemory);

    // Update descriptor set
    std::vector<VkDescriptorBufferInfo> bufferInfos = {
        {inputBuffer, 0, input_size},
        {weightBuffer, 0, weight_size},
        {config.bias ? biasBuffer : uniformBuffer, 0, config.bias ? bias_size : sizeof(LinearParams)}, // bias or dummy
        {outputBuffer, 0, output_size},
        {uniformBuffer, 0, sizeof(LinearParams)}
    };

    std::vector<VkWriteDescriptorSet> descriptorWrites;
    for (size_t i = 0; i < bufferInfos.size(); ++i) {
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptorSet;
        write.dstBinding = static_cast<uint32_t>(i);
        write.descriptorCount = 1;
        write.descriptorType = (i == 4) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &bufferInfos[i];
        descriptorWrites.push_back(write);
    }

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);

    // Execute compute shader
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    uint32_t workgroupCount = (total_elements + 63) / 64; // 64 is workgroup size from shader
    vkCmdDispatch(commandBuffer, workgroupCount, 1, 1);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    if (vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        throw std::runtime_error("failed to submit linear compute command buffer!");
    }
    if (vkQueueWaitIdle(computeQueue) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        throw std::runtime_error("failed to wait for linear queue idle!");
    }

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);

    // Copy result back
    VkBuffer outputStaging;
    VkDeviceMemory outputStagingMemory;
    createBuffer(output_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 outputStaging, outputStagingMemory);
    stagingBuffers.push_back({outputStaging, outputStagingMemory});

    copyBuffer(outputBuffer, outputStaging, output_size);

    void* outputData;
    vkMapMemory(device, outputStagingMemory, 0, output_size, 0, &outputData);
    memcpy(output.data.data(), outputData, output_size);
    vkUnmapMemory(device, outputStagingMemory);

    return output;
}

std::pair<Tensor, std::vector<Tensor>> LinearVulkan::backward(const Tensor& input, const Tensor& output_grad) {
    // For now, implement CPU backward pass
    // In a full implementation, this would use Vulkan compute shaders

    size_t batch_size = input.data.size() / config.in_features;

    // Gradient with respect to input: output_grad @ weights.T
    Tensor input_grad;
    input_grad.shape = input.shape;
    input_grad.data.resize(input.data.size(), 0.0f);

    // Gradient with respect to weights: input.T @ output_grad
    Tensor weight_grad;
    weight_grad.shape = {config.out_features, config.in_features};
    weight_grad.data.resize(config.out_features * config.in_features, 0.0f);

    // Gradient with respect to bias: sum(output_grad, axis=0)
    Tensor bias_grad;
    if (config.bias) {
        bias_grad.shape = {config.out_features};
        bias_grad.data.resize(config.out_features, 0.0f);
    }

    // Simple identity weights for demo (same as forward)
    std::vector<float> weights(config.out_features * config.in_features, 0.0f);
    for (size_t i = 0; i < std::min(static_cast<size_t>(config.in_features), static_cast<size_t>(config.out_features)); ++i) {
        weights[i * config.in_features + i] = 1.0f;
    }

    // Compute gradients
    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t out = 0; out < static_cast<size_t>(config.out_features); ++out) {
            float out_grad = output_grad.data[batch * config.out_features + out];

            // Input gradient: sum over output dimension
            for (size_t in = 0; in < static_cast<size_t>(config.in_features); ++in) {
                input_grad.data[batch * config.in_features + in] += out_grad * weights[out * config.in_features + in];
            }

            // Weight gradient
            for (size_t in = 0; in < static_cast<size_t>(config.in_features); ++in) {
                weight_grad.data[out * config.in_features + in] += input.data[batch * config.in_features + in] * out_grad;
            }

            // Bias gradient
            if (config.bias) {
                bias_grad.data[out] += out_grad;
            }
        }
    }

    std::vector<Tensor> param_grads;
    param_grads.push_back(weight_grad);
    if (config.bias) {
        param_grads.push_back(bias_grad);
    }

    return {input_grad, param_grads};
}

std::unordered_map<std::string, Tensor> LinearVulkan::get_parameters() {
    std::unordered_map<std::string, Tensor> params;
    // Return any cached CPU-side parameter tensors if present
    if (!weight_.data.empty()) {
        params["weight"] = weight_;
    } else if (weightBuffer != VK_NULL_HANDLE) {
        // Read weight buffer from device to host via staging buffer
        VkDeviceSize weight_size = static_cast<VkDeviceSize>(config.out_features) * config.in_features * sizeof(float);
        VkBuffer stagingBuf;
        VkDeviceMemory stagingMem;
        createBuffer(weight_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuf, stagingMem);
        stagingBuffers.push_back({stagingBuf, stagingMem});

        // Copy device weightBuffer -> stagingBuf
        copyBuffer(weightBuffer, stagingBuf, weight_size);

        void* mapped;
        vkMapMemory(device, stagingMem, 0, weight_size, 0, &mapped);
        Tensor w;
        w.shape = {config.out_features, config.in_features};
        w.data.resize(static_cast<size_t>(config.out_features) * config.in_features);
        memcpy(w.data.data(), mapped, static_cast<size_t>(weight_size));
        vkUnmapMemory(device, stagingMem);
        params["weight"] = std::move(w);
    }

    if (config.bias) {
        if (!bias_.data.empty()) {
            params["bias"] = bias_;
        } else if (biasBuffer != VK_NULL_HANDLE) {
            VkDeviceSize bias_size = static_cast<VkDeviceSize>(config.out_features) * sizeof(float);
            VkBuffer stagingBuf;
            VkDeviceMemory stagingMem;
            createBuffer(bias_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuf, stagingMem);
            stagingBuffers.push_back({stagingBuf, stagingMem});

            copyBuffer(biasBuffer, stagingBuf, bias_size);

            void* mapped;
            vkMapMemory(device, stagingMem, 0, bias_size, 0, &mapped);
            Tensor b;
            b.shape = {config.out_features};
            b.data.resize(static_cast<size_t>(config.out_features));
            memcpy(b.data.data(), mapped, static_cast<size_t>(bias_size));
            vkUnmapMemory(device, stagingMem);
            params["bias"] = std::move(b);
        }
    }

    return params;
}

void LinearVulkan::set_parameters(const std::unordered_map<std::string, Tensor>& params) {
    // Update CPU-side cached tensors and push to device buffers if present
    auto itW = params.find("weight");
    if (itW != params.end()) {
        weight_ = itW->second;

        if (device != VK_NULL_HANDLE) {
            VkDeviceSize weight_size = static_cast<VkDeviceSize>(weight_.data.size() * sizeof(float));
            if (weightBuffer == VK_NULL_HANDLE) {
                createBuffer(weight_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, weightBuffer, weightBufferMemory);
            }
            if (weightBuffer != VK_NULL_HANDLE) {
                VkBuffer stagingBuf;
                VkDeviceMemory stagingMem;
                createBuffer(weight_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuf, stagingMem);
                stagingBuffers.push_back({stagingBuf, stagingMem});

                void* mapped;
                vkMapMemory(device, stagingMem, 0, weight_size, 0, &mapped);
                memcpy(mapped, weight_.data.data(), static_cast<size_t>(weight_size));
                vkUnmapMemory(device, stagingMem);

                // Copy staging -> device
                copyBuffer(stagingBuf, weightBuffer, weight_size);
            }
        }
    }

    auto itB = params.find("bias");
    if (itB != params.end() && config.bias) {
        bias_ = itB->second;

        if (device != VK_NULL_HANDLE) {
            VkDeviceSize bias_size = static_cast<VkDeviceSize>(bias_.data.size() * sizeof(float));
            if (biasBuffer == VK_NULL_HANDLE) {
                createBuffer(bias_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, biasBuffer, biasBufferMemory);
            }
            if (biasBuffer != VK_NULL_HANDLE) {
                VkBuffer stagingBuf;
                VkDeviceMemory stagingMem;
                createBuffer(bias_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuf, stagingMem);
                stagingBuffers.push_back({stagingBuf, stagingMem});

                void* mapped;
                vkMapMemory(device, stagingMem, 0, bias_size, 0, &mapped);
                memcpy(mapped, bias_.data.data(), static_cast<size_t>(bias_size));
                vkUnmapMemory(device, stagingMem);

                copyBuffer(stagingBuf, biasBuffer, bias_size);
            }
        }
    }
}