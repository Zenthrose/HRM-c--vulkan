#include "rms_norm.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cmath>
#include <limits>

RMSNormVulkan::RMSNormVulkan(const RMSNormConfig& config, VkPhysicalDevice physicalDevice, VkDevice device, VkQueue computeQueue, uint32_t computeQueueFamilyIndex, VkCommandPool commandPool)
    : config(config), physicalDevice(physicalDevice), device(device), computeQueue(computeQueue), computeQueueFamilyIndex(computeQueueFamilyIndex), commandPool(commandPool),
      pipeline(VK_NULL_HANDLE), pipelineLayout(VK_NULL_HANDLE), descriptorSetLayout(VK_NULL_HANDLE), descriptorPool(VK_NULL_HANDLE) {
    std::cout << "Initializing RMSNormVulkan layer..." << std::endl;
    init_vulkan_objects();
}

RMSNormVulkan::~RMSNormVulkan() {
    std::cout << "Destroying RMSNormVulkan layer..." << std::endl;

    // Check if device is still valid before cleanup
    if (device == VK_NULL_HANDLE) {
        return; // Device already destroyed, skip cleanup
    }

    vkDestroyBuffer(device, inputBuffer, nullptr);
    vkFreeMemory(device, inputBufferMemory, nullptr);
    vkDestroyBuffer(device, outputBuffer, nullptr);
    vkFreeMemory(device, outputBufferMemory, nullptr);
    vkDestroyBuffer(device, uniformBuffer, nullptr);
    vkFreeMemory(device, uniformBufferMemory, nullptr);
    
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

Tensor RMSNormVulkan::forward(const Tensor& input) {
    
    // Map uniform buffer
    struct UniformData {
        uint32_t seq_len;
        uint32_t hidden_size;
        float variance_epsilon;
    } uniformData = {
        config.seq_len,
        config.hidden_size,
        config.variance_epsilon
    };

    void* uniformMapped;
    vkMapMemory(device, uniformBufferMemory, 0, sizeof(UniformData), 0, &uniformMapped);
    memcpy(uniformMapped, &uniformData, sizeof(UniformData));
    vkUnmapMemory(device, uniformBufferMemory);

    // Assume input is [batch, seq, hidden_size]
    uint32_t batch_size = input.data.size() / (config.seq_len * config.hidden_size);
    VkDeviceSize size = input.data.size() * sizeof(float);

    // Transfer input
    VkBuffer inputStagingBuffer;
    VkDeviceMemory inputStagingBufferMemory;
    createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, inputStagingBuffer, inputStagingBufferMemory);
    void* inputData;
    vkMapMemory(device, inputStagingBufferMemory, 0, size, 0, &inputData);
    memcpy(inputData, input.data.data(), size);
    vkUnmapMemory(device, inputStagingBufferMemory);
    copyBuffer(inputStagingBuffer, inputBuffer, size);
    vkDestroyBuffer(device, inputStagingBuffer, nullptr);
    vkFreeMemory(device, inputStagingBufferMemory, nullptr);

    // Dispatch
    VkCommandBufferAllocateInfo cmdBufAllocInfo{};
    cmdBufAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdBufAllocInfo.commandPool = commandPool;
    cmdBufAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufAllocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &cmdBufAllocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    uint32_t groupCountX = (batch_size * config.seq_len + 63) / 64; // One group per sequence
    vkCmdDispatch(commandBuffer, groupCountX, 1, 1);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    if (vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        throw std::runtime_error("failed to submit rms_norm compute command buffer!");
    }
    if (vkQueueWaitIdle(computeQueue) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        throw std::runtime_error("failed to wait for rms_norm queue idle!");
    }
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);

    // Transfer output
    Tensor output;
    output.shape = input.shape;
    output.data.resize(size / sizeof(float));

    VkBuffer outStagingBuffer;
    VkDeviceMemory outStagingBufferMemory;
    createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, outStagingBuffer, outStagingBufferMemory);
    copyBuffer(outputBuffer, outStagingBuffer, size);
    void* outData;
    vkMapMemory(device, outStagingBufferMemory, 0, size, 0, &outData);
    memcpy(output.data.data(), outData, size);
    vkUnmapMemory(device, outStagingBufferMemory);
    vkDestroyBuffer(device, outStagingBuffer, nullptr);
    vkFreeMemory(device, outStagingBufferMemory, nullptr);
    
    return output;
}

std::pair<Tensor, Tensor> RMSNormVulkan::backward(const Tensor& input, const Tensor& output_grad) {
    // RMS normalization backward pass
    // output = input / sqrt(mean(input^2) + eps)
    // dL/dinput = dL/doutput * d(output)/d(input)

    size_t batch_size = input.shape[0];
    size_t seq_len = input.shape[1];
    size_t hidden_size = config.hidden_size;

    Tensor input_grad;
    input_grad.shape = input.shape;
    input_grad.data.resize(input.data.size(), 0.0f);

    Tensor weight_grad; // RMS norm has learnable weight parameter
    if (hidden_size > std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error("hidden_size exceeds unsigned int maximum");
    }
    weight_grad.shape = {static_cast<unsigned int>(hidden_size)};
    weight_grad.data.resize(hidden_size, 0.0f);

    // Compute backward pass for each sample in the batch
    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t seq = 0; seq < seq_len; ++seq) {
            // Compute RMS for this sample
            float sum_squares = 0.0f;
            for (size_t h = 0; h < hidden_size; ++h) {
                size_t idx = (batch * seq_len + seq) * hidden_size + h;
                float val = input.data[idx];
                sum_squares += val * val;
            }
            float rms = sqrtf(sum_squares / hidden_size + config.variance_epsilon);

            // Compute gradients
            for (size_t h = 0; h < hidden_size; ++h) {
                size_t idx = (batch * seq_len + seq) * hidden_size + h;
                float input_val = input.data[idx];
                float out_grad = output_grad.data[idx];

                // Derivative of RMS norm: dL/dx = dL/dy * (1/rms) - dL/dy * x * d(rms)/dx / rms^2
                // Simplified: dL/dx = dL/dy / rms
                input_grad.data[idx] = out_grad / rms;

                // Accumulate weight gradients (assuming weight=1 for RMS norm)
                weight_grad.data[h] += out_grad * (input_val / rms);
            }
        }
    }

    return {input_grad, weight_grad};
}

void RMSNormVulkan::init_vulkan_objects() {
    createComputePipeline();

    uint32_t batch_size = 1;
    uint32_t seq_len = 256;
    uint32_t hidden_size = 128;
    VkDeviceSize size = batch_size * seq_len * hidden_size * sizeof(float);

    VkBufferUsageFlags storage_buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags memory_properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    createBuffer(size, storage_buffer_usage, memory_properties, inputBuffer, inputBufferMemory);
    createBuffer(size, storage_buffer_usage, memory_properties, outputBuffer, outputBufferMemory);
    createBuffer(sizeof(uint32_t) * 2 + sizeof(float), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffer, uniformBufferMemory);
    
    // Descriptor pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}, // input, output
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}
    };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;

    vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);

    // Descriptor set layout
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // input
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // output
        {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}  // config
    };

    VkDescriptorSetLayoutCreateInfo layoutCI{};
    layoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCI.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutCI.pBindings = bindings.data();

    vkCreateDescriptorSetLayout(device, &layoutCI, nullptr, &descriptorSetLayout);

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet);

    // Update descriptor set
    VkDescriptorBufferInfo input_info = {inputBuffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo output_info = {outputBuffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo uniform_info = {uniformBuffer, 0, VK_WHOLE_SIZE};

    std::vector<VkWriteDescriptorSet> descriptorWrites(3);
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &input_info;

    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = descriptorSet;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pBufferInfo = &output_info;

    descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[2].dstSet = descriptorSet;
    descriptorWrites[2].dstBinding = 2;
    descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[2].descriptorCount = 1;
    descriptorWrites[2].pBufferInfo = &uniform_info;

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
}

std::vector<char> RMSNormVulkan::readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();
    return buffer;
}

void RMSNormVulkan::createComputePipeline() {
    auto shaderCode = readFile("shaders/rms_norm.spv");

    VkShaderModuleCreateInfo shaderModuleCI{};
    shaderModuleCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCI.codeSize = shaderCode.size();
    shaderModuleCI.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());

    VkShaderModule computeShaderModule;
    vkCreateShaderModule(device, &shaderModuleCI, nullptr, &computeShaderModule);

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{};
    descriptorSetLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    descriptorSetLayoutCI.bindingCount = static_cast<uint32_t>(bindings.size());
    descriptorSetLayoutCI.pBindings = bindings.data();

    vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayout);

    VkPipelineLayoutCreateInfo pipelineLayoutCI{};
    pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCI.setLayoutCount = 1;
    pipelineLayoutCI.pSetLayouts = &descriptorSetLayout;

    vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayout);

    VkComputePipelineCreateInfo pipelineCI{};
    pipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCI.layout = pipelineLayout;
    pipelineCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineCI.stage.module = computeShaderModule;
    pipelineCI.stage.pName = "main";

    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &pipeline);

    vkDestroyShaderModule(device, computeShaderModule, nullptr);
}

uint32_t RMSNormVulkan::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void RMSNormVulkan::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory);
    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

void RMSNormVulkan::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
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
        throw std::runtime_error("failed to submit rms_norm copy command buffer!");
    }
    if (vkQueueWaitIdle(computeQueue) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        throw std::runtime_error("failed to wait for rms_norm copy queue idle!");
    }

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}
