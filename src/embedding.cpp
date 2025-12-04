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
    if (!device) {
        throw std::runtime_error("Vulkan device not available - cannot initialize EmbeddingVulkan layer");
    }
    init_vulkan_objects();
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

    vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(computeQueue);
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);

    // Transfer output
    Tensor output;
    output.shape = {static_cast<uint32_t>(input.size()), config.embedding_dim};
    output.data.resize(input.size() * config.embedding_dim);

    VkBuffer outStagingBuffer;
    VkDeviceMemory outStagingBufferMemory;
    createBuffer(output_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, outStagingBuffer, outStagingBufferMemory);
    copyBuffer(outputBuffer, outStagingBuffer, output_size);
    void* outData;
    vkMapMemory(device, outStagingBufferMemory, 0, output_size, 0, &outData);
    memcpy(output.data.data(), outData, output_size);
    vkUnmapMemory(device, outStagingBufferMemory);
    vkDestroyBuffer(device, outStagingBuffer, nullptr);
    vkFreeMemory(device, outStagingBufferMemory, nullptr);
    
    return output;
}

void EmbeddingVulkan::init_vulkan_objects() {
    createComputePipeline();

    // Create buffers with dummy sizes (will be resized in forward)
    VkDeviceSize dummy_size = 1024 * sizeof(float);
    VkBufferUsageFlags storage_buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags memory_properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    createBuffer(dummy_size, storage_buffer_usage, memory_properties, inputBuffer, inputBufferMemory);
    createBuffer(dummy_size, storage_buffer_usage, memory_properties, weightBuffer, weightBufferMemory);
    createBuffer(dummy_size, storage_buffer_usage, memory_properties, outputBuffer, outputBufferMemory);
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

    vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(computeQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}