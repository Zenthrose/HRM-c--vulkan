#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#ifndef NO_VULKAN
#include <vulkan/vulkan.h>
#endif

#include "attention.hpp" // For Tensor

struct LinearConfig {
    uint32_t in_features;
    uint32_t out_features;
    bool bias;
};

#ifndef NO_VULKAN
// Linear layer parameters for Vulkan uniform buffer
struct LinearParams {
    uint32_t batch_size;
    uint32_t in_features;
    uint32_t out_features;
    uint32_t has_bias;
};

class LinearVulkan {
public:
    // Parameter storage for training
    Tensor weight_;
    Tensor bias_;
public:
    LinearVulkan(const LinearConfig& config, VkPhysicalDevice physicalDevice, VkDevice device, VkQueue computeQueue, uint32_t computeQueueFamilyIndex, VkCommandPool commandPool);
    ~LinearVulkan();

    Tensor forward(const Tensor& input);
    std::pair<Tensor, std::vector<Tensor>> backward(const Tensor& input, const Tensor& output_grad);

    // Parameter access for training
    std::unordered_map<std::string, Tensor> get_parameters();
    void set_parameters(const std::unordered_map<std::string, Tensor>& params);

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
#endif