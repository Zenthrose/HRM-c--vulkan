#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>

#include "attention.hpp" // For Tensor

struct LinearConfig {
    uint32_t in_features;
    uint32_t out_features;
    bool bias;
};

class LinearVulkan {
public:
    LinearVulkan(const LinearConfig& config, VkPhysicalDevice physicalDevice, VkDevice device, VkQueue computeQueue, uint32_t computeQueueFamilyIndex, VkCommandPool commandPool);
    ~LinearVulkan();

    Tensor forward(const Tensor& input);

private:
    void init_vulkan_objects();

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

    VkBuffer inputBuffer, weightBuffer, biasBuffer, outputBuffer;
    VkDeviceMemory inputBufferMemory, weightBufferMemory, biasBufferMemory, outputBufferMemory;

    VkBuffer uniformBuffer;
    VkDeviceMemory uniformBufferMemory;

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    static std::vector<char> readFile(const std::string& filename);
    void createComputePipeline();
};