#pragma once

#ifndef NO_VULKAN
#include <vulkan/vulkan.h>
#endif
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <cstdint>

// A simple tensor representation for now
struct Tensor {
    std::vector<float> data;
    std::vector<uint32_t> shape;
};

struct CosSin {
    std::vector<float> cos;
    std::vector<float> sin;
};

struct AttentionConfig {
    uint32_t batch_size;
    uint32_t seq_len;
    uint32_t head_dim;
    uint32_t num_heads;
    uint32_t num_key_value_heads;
    bool causal;
};

#ifndef NO_VULKAN
class AttentionVulkan {
public:
    AttentionVulkan(const AttentionConfig& config, VkPhysicalDevice physicalDevice, VkDevice device, VkQueue computeQueue, uint32_t computeQueueFamilyIndex, VkCommandPool commandPool);
    ~AttentionVulkan();

    Tensor forward(const Tensor& hidden_states, const CosSin& cos_sin);
    std::pair<Tensor, std::vector<Tensor>> backward(const Tensor& hidden_states, const Tensor& output_grad);

private:
    void init_vulkan_objects();

    AttentionConfig config;
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

    // Buffers for Q, K, V, and Output
    VkBuffer queryBuffer, keyBuffer, valueBuffer, outputBuffer;
    VkDeviceMemory queryBufferMemory, keyBufferMemory, valueBufferMemory, outputBufferMemory;

    // Uniform buffer
    VkBuffer uniformBuffer;
    VkDeviceMemory uniformBufferMemory;

    // Track staging buffers for cleanup
    std::vector<std::pair<VkBuffer, VkDeviceMemory>> stagingBuffers;

    // Helper function to create a buffer
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);

    // Helper function to find memory type
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

    // Helper function to load a shader
    static std::vector<char> readFile(const std::string& filename);
    void createComputePipeline();



};
#endif

// CPU reference implementations for testing and numerical checks
std::pair<Tensor, std::vector<Tensor>> cpu_attention_backward(const Tensor& hidden_states, const Tensor& output_grad, const AttentionConfig& config);