#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>

#include "attention.hpp" // For Tensor

struct EmbeddingConfig {
    uint32_t num_embeddings;
    uint32_t embedding_dim;
    uint32_t seq_len;
};

class EmbeddingVulkan {
public:
    EmbeddingVulkan(const EmbeddingConfig& config, VkPhysicalDevice physicalDevice, VkDevice device, VkQueue computeQueue, uint32_t computeQueueFamilyIndex, VkCommandPool commandPool);
    ~EmbeddingVulkan();

    Tensor forward(const std::vector<uint32_t>& input);

private:
    void init_vulkan_objects();

    EmbeddingConfig config;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue computeQueue;
    uint32_t computeQueueFamilyIndex;
    VkCommandPool commandPool;

    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;

    VkBuffer inputBuffer, outputBuffer;
    VkDeviceMemory inputBufferMemory, outputBufferMemory;

    std::vector<VkBuffer> weightBuffers;
    std::vector<VkDeviceMemory> weightBufferMemories;
    uint32_t chunk_size = 10000;
    uint32_t num_chunks;

    VkBuffer uniformBuffer;
    VkDeviceMemory uniformBufferMemory;

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    static std::vector<char> readFile(const std::string& filename);
    void createComputePipeline();
};