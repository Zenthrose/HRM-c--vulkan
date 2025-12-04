#include "vulkan_resource_manager.hpp"
#include <iostream>
#include <cstring>
#include <stdexcept>

VulkanResourceManager::VulkanResourceManager(VkDevice device, VkPhysicalDevice physicalDevice, VkQueue computeQueue, uint32_t computeQueueFamilyIndex)
    : device(device), physicalDevice(physicalDevice), computeQueue(computeQueue), computeQueueFamilyIndex(computeQueueFamilyIndex), commandPool(VK_NULL_HANDLE) {
    createCommandPool();
}

VulkanResourceManager::~VulkanResourceManager() {
    // Destroy all tracked buffers in reverse order
    for (auto it = trackedBuffers.rbegin(); it != trackedBuffers.rend(); ++it) {
        if (it->buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, it->buffer, nullptr);
        }
        if (it->memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, it->memory, nullptr);
        }
    }
    trackedBuffers.clear();

    destroyCommandPool();
}

void VulkanResourceManager::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
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

    // Track the buffer for cleanup
    trackedBuffers.push_back({buffer, bufferMemory});
}

void VulkanResourceManager::destroyBuffer(VkBuffer buffer, VkDeviceMemory memory) {
    // Remove from tracked buffers
    trackedBuffers.erase(
        std::remove_if(trackedBuffers.begin(), trackedBuffers.end(),
            [buffer, memory](const TrackedBuffer& tb) {
                return tb.buffer == buffer && tb.memory == memory;
            }),
        trackedBuffers.end()
    );

    // Destroy immediately
    if (buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buffer, nullptr);
    }
    if (memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, memory, nullptr);
    }
}

std::unique_ptr<VulkanResourceManager::StagingBuffer> VulkanResourceManager::createStagingBuffer(VkDeviceSize size, VkBufferUsageFlags usage) {
    return std::unique_ptr<StagingBuffer>(new StagingBuffer(physicalDevice, device, size, usage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
}

void VulkanResourceManager::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
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

uint32_t VulkanResourceManager::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void VulkanResourceManager::createCommandPool() {
    // TODO: Need queue family index parameter
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = 0; // TODO: Pass proper queue family index

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }
}

void VulkanResourceManager::destroyCommandPool() {
    if (commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device, commandPool, nullptr);
        commandPool = VK_NULL_HANDLE;
    }
}

// StagingBuffer implementation
VulkanResourceManager::StagingBuffer::StagingBuffer(VkPhysicalDevice physDev, VkDevice dev, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties)
    : physicalDevice(physDev), device(dev), buffer(VK_NULL_HANDLE), memory(VK_NULL_HANDLE), mappedData(nullptr) {

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create staging buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        vkDestroyBuffer(device, buffer, nullptr);
        throw std::runtime_error("failed to allocate staging buffer memory!");
    }

    vkBindBufferMemory(device, buffer, memory, 0);
}

VulkanResourceManager::StagingBuffer::~StagingBuffer() {
    unmap();
    if (buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buffer, nullptr);
    }
    if (memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, memory, nullptr);
    }
}

void VulkanResourceManager::StagingBuffer::map() {
    if (mappedData == nullptr) {
        vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mappedData);
    }
}

void VulkanResourceManager::StagingBuffer::unmap() {
    if (mappedData != nullptr) {
        vkUnmapMemory(device, memory);
        mappedData = nullptr;
    }
}

void VulkanResourceManager::StagingBuffer::copyFrom(const void* data, VkDeviceSize size) {
    map();
    memcpy(mappedData, data, size);
    unmap();
}

void VulkanResourceManager::StagingBuffer::copyTo(void* data, VkDeviceSize size) {
    map();
    memcpy(data, mappedData, size);
    unmap();
}

uint32_t VulkanResourceManager::StagingBuffer::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}