#pragma once

#ifndef NO_VULKAN
#include <vulkan/vulkan.h>
#endif
#include <vector>
#include <string>

#include "attention.hpp" // For Tensor
#include <unordered_map>

struct EmbeddingConfig {
    uint32_t num_embeddings;
    uint32_t embedding_dim;
    uint32_t seq_len;
};

#ifndef NO_VULKAN
class EmbeddingVulkan {
public:
    EmbeddingVulkan(const EmbeddingConfig& config, VkPhysicalDevice physicalDevice, VkDevice device, VkQueue computeQueue, uint32_t computeQueueFamilyIndex, VkCommandPool commandPool);
    ~EmbeddingVulkan();

    Tensor forward(const std::vector<uint32_t>& input);
    std::vector<Tensor> backward(const std::vector<uint32_t>& input, const Tensor& output_grad);
    // Parameter access for training
    std::unordered_map<std::string, Tensor> get_parameters();
    void set_parameters(const std::unordered_map<std::string, Tensor>& params);

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
    std::string getShaderPath(const std::string& shaderName);
};
#endif