#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <memory>
#include <algorithm>

/**
 * Vulkan Resource Manager
 * 
 * Manages the lifecycle of all Vulkan objects to ensure proper cleanup
 * and prevent resource leaks. Tracks all created objects and destroys
 * them in the correct order during shutdown.
 */
class VulkanResourceManager {
public:
    VulkanResourceManager(VkDevice device, VkPhysicalDevice physicalDevice, VkQueue computeQueue, uint32_t computeQueueFamilyIndex);
    ~VulkanResourceManager();

    // Buffer management
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void destroyBuffer(VkBuffer buffer, VkDeviceMemory memory);

    // Staging buffer helpers
    class StagingBuffer {
    public:
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        void* mappedData = nullptr;

        StagingBuffer(VkPhysicalDevice physDev, VkDevice dev, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
        ~StagingBuffer();

        void map();
        void unmap();
        void copyFrom(const void* data, VkDeviceSize size);
        void copyTo(void* data, VkDeviceSize size);

    private:
        VkPhysicalDevice physicalDevice;
        static uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);
    };

    std::unique_ptr<StagingBuffer> createStagingBuffer(VkDeviceSize size, VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

    // Memory type finding
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

private:
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    VkQueue computeQueue;
    uint32_t computeQueueFamilyIndex;
    VkCommandPool commandPool;

    // Track all created objects for proper cleanup
    struct TrackedBuffer {
        VkBuffer buffer;
        VkDeviceMemory memory;
    };
    std::vector<TrackedBuffer> trackedBuffers;

    // Helper functions
    void createCommandPool();
    void destroyCommandPool();

private:
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    VkCommandPool commandPool;

    // Track all created objects for proper cleanup
    struct TrackedBuffer {
        VkBuffer buffer;
        VkDeviceMemory memory;
    };
    std::vector<TrackedBuffer> trackedBuffers;

    // Helper functions
    void createCommandPool();
    void destroyCommandPool();
};