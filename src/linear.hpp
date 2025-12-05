#pragma once

#include <vector>
#include <string>
#include <vulkan/vulkan.h>

#include "attention.hpp" // For Tensor

struct LinearConfig {
    uint32_t in_features;
    uint32_t out_features;
    bool bias;
};

struct LinearParams {
    uint32_t batch_seq;
    uint32_t in_features;
    uint32_t out_features;
    uint32_t has_bias;
};

class LinearVulkan {
public:
    LinearVulkan(const LinearConfig& config, VkPhysicalDevice physicalDevice, VkDevice device, VkQueue computeQueue, uint32_t computeQueueFamilyIndex, VkCommandPool commandPool);
    ~LinearVulkan();

    Tensor forward(const Tensor& input);

private:
    void init_vulkan_objects();
    void createComputePipeline();
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    LinearConfig config;
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

    VkBuffer inputBuffer, weightBuffer, biasBuffer, outputBuffer, uniformBuffer;
    VkDeviceMemory inputBufferMemory, weightBufferMemory, biasBufferMemory, outputBufferMemory, uniformBufferMemory;

    std::vector<std::pair<VkBuffer, VkDeviceMemory>> stagingBuffers;
};