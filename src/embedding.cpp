#include "embedding.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstring>

EmbeddingVulkan::EmbeddingVulkan(const EmbeddingConfig& config, VkPhysicalDevice physicalDevice, VkDevice device, VkQueue computeQueue, uint32_t computeQueueFamilyIndex, VkCommandPool commandPool)
    : config(config), physicalDevice(physicalDevice), device(device), computeQueue(computeQueue), computeQueueFamilyIndex(computeQueueFamilyIndex), commandPool(commandPool),
      pipeline(VK_NULL_HANDLE), pipelineLayout(VK_NULL_HANDLE), descriptorSetLayout(VK_NULL_HANDLE), descriptorPool(VK_NULL_HANDLE) {
    std::cout << "Initializing EmbeddingVulkan layer..." << std::endl;
    if (device) {
        init_vulkan_objects();
    } else {
        std::cout << "Vulkan device not available - using CPU fallback for Embedding layer" << std::endl;
    }
}

EmbeddingVulkan::~EmbeddingVulkan() {
    std::cout << "Destroying EmbeddingVulkan layer..." << std::endl;

    vkDestroyBuffer(device, inputBuffer, nullptr);
    vkFreeMemory(device, inputBufferMemory, nullptr);
    vkDestroyBuffer(device, weightBuffer, nullptr);
    vkFreeMemory(device, weightBufferMemory, nullptr);
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

Tensor EmbeddingVulkan::forward(const std::vector<uint32_t>& input) {
    std::cout << "Performing forward pass in EmbeddingVulkan..." << std::endl;

    if (!device) {
        // CPU fallback implementation
        Tensor raw_output;
        raw_output.shape = {static_cast<unsigned int>(input.size()), config.embedding_dim};
        raw_output.data.resize(input.size() * config.embedding_dim);

        // Simple embedding lookup (random values for demo)
        for (size_t i = 0; i < input.size(); ++i) {
            uint32_t token_id = input[i];
            for (size_t j = 0; j < config.embedding_dim; ++j) {
                // Simple hash-based pseudo-random embedding
                raw_output.data[i * config.embedding_dim + j] = static_cast<float>((token_id * 31 + j * 17) % 1000) / 1000.0f - 0.5f;
            }
        }

        // Average over sequence dimension
        int batch_size = 2;  // Hardcoded for now
        int seq_len = 512;   // Hardcoded for now
        uint32_t embedding_dim = static_cast<uint32_t>(config.embedding_dim);
        Tensor output;
        output.data.resize(batch_size * embedding_dim, 0.0f);
        output.shape = {static_cast<uint32_t>(batch_size), embedding_dim};

        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < embedding_dim; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < seq_len; ++k) {
                    sum += raw_output.data[(i * seq_len + k) * embedding_dim + j];
                }
                output.data[i * embedding_dim + j] = sum / seq_len;
            }
        }

        return output;
    }

    // Vulkan implementation
    
    // Update uniform buffer
    struct UniformData {
        uint32_t seq_len;
        uint32_t embedding_dim;
    } uniformData = {
        static_cast<uint32_t>(input.size()),
        config.embedding_dim
    };
    
    void* uniformMapped;
    vkMapMemory(device, uniformBufferMemory, 0, sizeof(UniformData), 0, &uniformMapped);
    memcpy(uniformMapped, &uniformData, sizeof(UniformData));
    vkUnmapMemory(device, uniformBufferMemory);

    VkDeviceSize input_size = input.size() * sizeof(uint32_t);
    VkDeviceSize weight_size = config.num_embeddings * config.embedding_dim * sizeof(float);
    VkDeviceSize output_size = input.size() * config.embedding_dim * sizeof(float);

    // Recreate buffers with correct sizes
    VkBufferUsageFlags storage_buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags memory_properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Recreate inputBuffer
    vkDestroyBuffer(device, inputBuffer, nullptr);
    vkFreeMemory(device, inputBufferMemory, nullptr);
    createBuffer(input_size, storage_buffer_usage, memory_properties, inputBuffer, inputBufferMemory);

    // Recreate weightBuffer
    vkDestroyBuffer(device, weightBuffer, nullptr);
    vkFreeMemory(device, weightBufferMemory, nullptr);
    createBuffer(weight_size, storage_buffer_usage, memory_properties, weightBuffer, weightBufferMemory);

    // Initialize weights with pseudo-random values
    VkBuffer weightStagingBuf;
    VkDeviceMemory weightStagingMem;
    createBuffer(weight_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, weightStagingBuf, weightStagingMem);

    void* weightDataPtr;
    vkMapMemory(device, weightStagingMem, 0, weight_size, 0, &weightDataPtr);
    float* weightPtr = static_cast<float*>(weightDataPtr);
    for (uint32_t i = 0; i < config.num_embeddings; ++i) {
        for (uint32_t j = 0; j < config.embedding_dim; ++j) {
            uint32_t idx = i * config.embedding_dim + j;
            weightPtr[idx] = static_cast<float>((i * 31 + j * 17) % 1000) / 1000.0f - 0.5f;
        }
    }
    vkUnmapMemory(device, weightStagingMem);

    copyBuffer(weightStagingBuf, weightBuffer, weight_size);
    vkDestroyBuffer(device, weightStagingBuf, nullptr);
    vkFreeMemory(device, weightStagingMem, nullptr);

    // Recreate outputBuffer
    vkDestroyBuffer(device, outputBuffer, nullptr);
    vkFreeMemory(device, outputBufferMemory, nullptr);
    createBuffer(output_size, storage_buffer_usage, memory_properties, outputBuffer, outputBufferMemory);

    // Transfer input
    VkBuffer inputStagingBuffer;
    VkDeviceMemory inputStagingBufferMemory;
    createBuffer(input_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, inputStagingBuffer, inputStagingBufferMemory);
    void* inputData;
    vkMapMemory(device, inputStagingBufferMemory, 0, input_size, 0, &inputData);
    memcpy(inputData, input.data(), input_size);
    vkUnmapMemory(device, inputStagingBufferMemory);
    copyBuffer(inputStagingBuffer, inputBuffer, input_size);
    vkDestroyBuffer(device, inputStagingBuffer, nullptr);
    vkFreeMemory(device, inputStagingBufferMemory, nullptr);

    // Update descriptors
    VkDescriptorBufferInfo inputBufferInfo{};
    inputBufferInfo.buffer = inputBuffer;
    inputBufferInfo.offset = 0;
    inputBufferInfo.range = input_size;

    VkDescriptorBufferInfo weightBufferInfo{};
    weightBufferInfo.buffer = weightBuffer;
    weightBufferInfo.offset = 0;
    weightBufferInfo.range = weight_size;

    VkDescriptorBufferInfo outputBufferInfo{};
    outputBufferInfo.buffer = outputBuffer;
    outputBufferInfo.offset = 0;
    outputBufferInfo.range = output_size;

    VkDescriptorBufferInfo uniformBufferInfo{};
    uniformBufferInfo.buffer = uniformBuffer;
    uniformBufferInfo.offset = 0;
    uniformBufferInfo.range = sizeof(UniformData);

    std::vector<VkWriteDescriptorSet> descriptorWrites = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &inputBufferInfo, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &weightBufferInfo, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &outputBufferInfo, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uniformBufferInfo, nullptr}
    };

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);

    // Transfer weights (initialize with random values for demo)
    VkBuffer weightStagingBuffer;
    VkDeviceMemory weightStagingBufferMemory;
    createBuffer(weight_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, weightStagingBuffer, weightStagingBufferMemory);
    std::vector<float> weights(config.num_embeddings * config.embedding_dim, 0.0f);
    // Initialize with small random values
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
    }
    void* weightData;
    vkMapMemory(device, weightStagingBufferMemory, 0, weight_size, 0, &weightData);
    memcpy(weightData, weights.data(), weight_size);
    vkUnmapMemory(device, weightStagingBufferMemory);
    copyBuffer(weightStagingBuffer, weightBuffer, weight_size);
    vkDestroyBuffer(device, weightStagingBuffer, nullptr);
    vkFreeMemory(device, weightStagingBufferMemory, nullptr);

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

    uint32_t groupCountX = (input.size() * config.embedding_dim + 63) / 64;
    vkCmdDispatch(commandBuffer, groupCountX, 1, 1);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    if (vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        throw std::runtime_error("failed to submit embedding compute command buffer!");
    }
    if (vkQueueWaitIdle(computeQueue) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        throw std::runtime_error("failed to wait for embedding queue idle!");
    }
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);

    // Transfer output
    Tensor raw_output;
    raw_output.shape = {static_cast<uint32_t>(input.size()), config.embedding_dim};
    raw_output.data.resize(input.size() * config.embedding_dim);

    VkBuffer outStagingBuffer;
    VkDeviceMemory outStagingBufferMemory;
    createBuffer(output_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, outStagingBuffer, outStagingBufferMemory);
    copyBuffer(outputBuffer, outStagingBuffer, output_size);
    void* outData;
    vkMapMemory(device, outStagingBufferMemory, 0, output_size, 0, &outData);
    memcpy(raw_output.data.data(), outData, output_size);
    vkUnmapMemory(device, outStagingBufferMemory);
    vkDestroyBuffer(device, outStagingBuffer, nullptr);
    vkFreeMemory(device, outStagingBufferMemory, nullptr);

    // Average over sequence dimension to match HRM batch input
    // Assuming input.size() = batch_size * seq_len
    uint32_t batch_size = 2;  // Hardcoded for now
    uint32_t seq_len = 512;   // Hardcoded for now
    uint32_t embedding_dim = static_cast<uint32_t>(config.embedding_dim);
    Tensor output;
    output.data.resize(batch_size * embedding_dim, 0.0f);
    output.shape = {batch_size, embedding_dim};

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < embedding_dim; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < seq_len; ++k) {
                sum += raw_output.data[(i * seq_len + k) * embedding_dim + j];
            }
            output.data[i * embedding_dim + j] = sum / seq_len;
        }
    }

    return output;
}

void EmbeddingVulkan::init_vulkan_objects() {
    createComputePipeline();

    // Create buffers with sufficient sizes
    VkDeviceSize buffer_size = 1024 * 1024; // 1MB, sufficient for initial operations
    VkBufferUsageFlags storage_buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags memory_properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    createBuffer(buffer_size, storage_buffer_usage, memory_properties, inputBuffer, inputBufferMemory);
    createBuffer(buffer_size, storage_buffer_usage, memory_properties, weightBuffer, weightBufferMemory);
    createBuffer(buffer_size, storage_buffer_usage, memory_properties, outputBuffer, outputBufferMemory);
    createBuffer(sizeof(uint32_t) * 2, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffer, uniformBufferMemory);
    
    // Descriptor pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3}, // input, weights, output
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
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // weights
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // output
        {3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}  // params
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
}

std::vector<char> EmbeddingVulkan::readFile(const std::string& filename) {
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

void EmbeddingVulkan::createComputePipeline() {
    auto shaderCode = readFile("shaders/embedding.spv");

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
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
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

uint32_t EmbeddingVulkan::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void EmbeddingVulkan::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        vkDestroyBuffer(device, buffer, nullptr);
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    if (vkBindBufferMemory(device, buffer, bufferMemory, 0) != VK_SUCCESS) {
        vkDestroyBuffer(device, buffer, nullptr);
        vkFreeMemory(device, bufferMemory, nullptr);
        throw std::runtime_error("failed to bind buffer memory!");
    }
}

void EmbeddingVulkan::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
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
        throw std::runtime_error("failed to submit embedding copy command buffer!");
    }
    if (vkQueueWaitIdle(computeQueue) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        throw std::runtime_error("failed to wait for embedding copy queue idle!");
    }

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}