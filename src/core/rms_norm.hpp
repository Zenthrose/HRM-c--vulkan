#pragma once

#ifndef NO_VULKAN
#include <vulkan/vulkan.h>
#endif
#include <vector>
#include <string>

#include "attention.hpp" // For Tensor

struct RMSNormConfig {
    uint32_t seq_len;
    uint32_t hidden_size;
    float variance_epsilon;
};

#ifndef NO_VULKAN
class RMSNormVulkan {
public:
    RMSNormVulkan(const RMSNormConfig& config, VkPhysicalDevice physicalDevice, VkDevice device, VkQueue computeQueue, uint32_t computeQueueFamilyIndex, VkCommandPool commandPool);
    ~RMSNormVulkan();

    Tensor forward(const Tensor& input);
    std::pair<Tensor, Tensor> backward(const Tensor& input, const Tensor& output_grad);

private:
    void init_vulkan_objects();

    RMSNormConfig config;
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

    VkBuffer uniformBuffer;
    VkDeviceMemory uniformBufferMemory;

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    static std::vector<char> readFile(const std::string& filename);
    void createComputePipeline();
};
#endif