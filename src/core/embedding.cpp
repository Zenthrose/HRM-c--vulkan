#include "embedding.hpp"
#include "../utils/cpu_compatibility.h"
#include "../vulkan/vulkan_compatibility.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <thread>
#include <algorithm>
#include <numeric>
#include <filesystem>

namespace fs = std::filesystem;

EmbeddingVulkan::EmbeddingVulkan(const EmbeddingConfig& config, VkPhysicalDevice physicalDevice, VkDevice device, VkQueue computeQueue, uint32_t computeQueueFamilyIndex, VkCommandPool commandPool)
    : config(config), physicalDevice(physicalDevice), device(device), computeQueue(computeQueue), computeQueueFamilyIndex(computeQueueFamilyIndex), commandPool(commandPool),
      pipeline(VK_NULL_HANDLE), pipelineLayout(VK_NULL_HANDLE), descriptorSetLayout(VK_NULL_HANDLE), descriptorPool(VK_NULL_HANDLE),
      num_chunks((config.num_embeddings + chunk_size - 1) / chunk_size) {
    std::cout << "Initializing EmbeddingVulkan layer..." << std::endl;
    if (device) {
        init_vulkan_objects();
    } else {
        std::cout << "Vulkan device not available - using CPU fallback for Embedding layer" << std::endl;
    }
}

EmbeddingVulkan::~EmbeddingVulkan() {
    std::cout << "Destroying EmbeddingVulkan layer..." << std::endl;

    // Only destroy Vulkan resources if device is valid
    if (device != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, inputBuffer, nullptr);
        vkFreeMemory(device, inputBufferMemory, nullptr);
        for (size_t i = 0; i < weightBuffers.size(); ++i) {
            vkDestroyBuffer(device, weightBuffers[i], nullptr);
            vkFreeMemory(device, weightBufferMemories[i], nullptr);
        }
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
}

Tensor EmbeddingVulkan::forward(const std::vector<uint32_t>& input) {

    if (!device) {
        // CPU fallback implementation with improved performance
        Tensor raw_output;
        raw_output.shape = {static_cast<unsigned int>(input.size()), config.embedding_dim};

        // Initialize output data
        raw_output.data.resize(input.size() * config.embedding_dim);

        // Multi-threaded embedding generation
        CpuFeatures cpu_features = CpuCompatibility::detectCpuFeatures();
        unsigned int num_threads = std::min(cpu_features.num_threads, static_cast<int>(input.size()));

        auto embedding_worker = [&](size_t start_idx, size_t end_idx) {
            for (size_t i = start_idx; i < end_idx; ++i) {
                uint32_t token_id = input[i];
                for (size_t j = 0; j < config.embedding_dim; ++j) {
                    // Improved hash-based pseudo-random embedding
                    uint32_t hash = token_id * 31 + static_cast<uint32_t>(j) * 17;
                    hash = (hash ^ (hash >> 15)) * 0x45d9f3b;
                    hash = (hash ^ (hash >> 15)) * 0x45d9f3b;
                    hash = hash ^ (hash >> 15);
                    raw_output.data[i * config.embedding_dim + j] = static_cast<float>(hash % 2000) / 1000.0f - 1.0f;
                }
            }
        };

        std::vector<std::thread> threads;
        size_t chunk_size = input.size() / num_threads;
        size_t remainder = input.size() % num_threads;

        size_t start = 0;
        for (unsigned int t = 0; t < num_threads; ++t) {
            size_t end = start + chunk_size + (t < remainder ? 1 : 0);
            threads.emplace_back(embedding_worker, start, end);
            start = end;
        }

        for (auto& thread : threads) {
            thread.join();
        }

        // Average over sequence dimension with better parameter detection
        // For now, assume reasonable defaults - this should be passed as parameters
        int batch_size = 1;  // Assume single batch for simplicity
        int seq_len = static_cast<int>(input.size());  // Use actual input size
        uint32_t embedding_dim = static_cast<uint32_t>(config.embedding_dim);

        Tensor output;
        output.data.resize(batch_size * embedding_dim, 0.0f);
        output.shape = {static_cast<uint32_t>(batch_size), embedding_dim};

        // Parallel averaging with SIMD acceleration
        auto averaging_worker = [&](size_t start_dim, size_t end_dim) {
            std::vector<float> temp_data(input.size());
            for (size_t j = start_dim; j < end_dim; ++j) {
                for (size_t k = 0; k < input.size(); ++k) {
                    temp_data[k] = raw_output.data[k * embedding_dim + j];
                }
                float sum = CpuCompatibility::vectorSum(temp_data.data(), input.size());
                output.data[j] = sum / static_cast<float>(input.size());
            }
        };

        threads.clear();
        chunk_size = embedding_dim / num_threads;
        remainder = embedding_dim % num_threads;
        start = 0;

        for (unsigned int t = 0; t < num_threads; ++t) {
            size_t end = start + chunk_size + (t < remainder ? 1 : 0);
            threads.emplace_back(averaging_worker, start, end);
            start = end;
        }

        for (auto& thread : threads) {
            thread.join();
        }

        return output;
    }

    // Vulkan implementation
    
    // Update uniform buffer
    struct UniformData {
        uint32_t seq_len;
        uint32_t embedding_dim;
        uint32_t chunk_size;
    } uniformData = {
        static_cast<uint32_t>(input.size()),
        config.embedding_dim,
        chunk_size
    };
    
    void* uniformMapped;
    vkMapMemory(device, uniformBufferMemory, 0, sizeof(UniformData), 0, &uniformMapped);
    memcpy(uniformMapped, &uniformData, sizeof(UniformData));
    vkUnmapMemory(device, uniformBufferMemory);

    VkDeviceSize input_size = input.size() * sizeof(uint32_t);
    VkDeviceSize output_size = input.size() * config.embedding_dim * sizeof(float);

    // Recreate buffers with correct sizes
    VkBufferUsageFlags storage_buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags memory_properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Recreate inputBuffer
    vkDestroyBuffer(device, inputBuffer, nullptr);
    vkFreeMemory(device, inputBufferMemory, nullptr);
    createBuffer(input_size, storage_buffer_usage, memory_properties, inputBuffer, inputBufferMemory);

    // Recreate weightBuffers
    for (size_t i = 0; i < weightBuffers.size(); ++i) {
        vkDestroyBuffer(device, weightBuffers[i], nullptr);
        vkFreeMemory(device, weightBufferMemories[i], nullptr);
    }
    weightBuffers.clear();
    weightBufferMemories.clear();
    for (uint32_t i = 0; i < num_chunks; ++i) {
        uint32_t chunk_vocab = (i == num_chunks - 1) ? (config.num_embeddings % chunk_size) : chunk_size;
        VkDeviceSize chunk_weight_size = chunk_vocab * config.embedding_dim * sizeof(float);
        VkBuffer buffer;
        VkDeviceMemory memory;
        createBuffer(chunk_weight_size, storage_buffer_usage, memory_properties, buffer, memory);
        weightBuffers.push_back(buffer);
        weightBufferMemories.push_back(memory);
    }

    // Initialize weights with pseudo-random values for each chunk
    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        uint32_t chunk_vocab = (chunk == num_chunks - 1) ? (config.num_embeddings % chunk_size) : chunk_size;
        VkDeviceSize chunk_weight_size = chunk_vocab * config.embedding_dim * sizeof(float);
        VkBuffer weightStagingBuf;
        VkDeviceMemory weightStagingMem;
        createBuffer(chunk_weight_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, weightStagingBuf, weightStagingMem);

        void* weightDataPtr;
        vkMapMemory(device, weightStagingMem, 0, chunk_weight_size, 0, &weightDataPtr);
        float* weightPtr = static_cast<float*>(weightDataPtr);
        for (uint32_t i = 0; i < chunk_vocab; ++i) {
            uint32_t global_i = chunk * chunk_size + i;
            for (uint32_t j = 0; j < config.embedding_dim; ++j) {
                uint32_t idx = i * config.embedding_dim + j;
                weightPtr[idx] = static_cast<float>((global_i * 31 + j * 17) % 1000) / 1000.0f - 0.5f;
            }
        }
        vkUnmapMemory(device, weightStagingMem);

        copyBuffer(weightStagingBuf, weightBuffers[chunk], chunk_weight_size);
        vkDestroyBuffer(device, weightStagingBuf, nullptr);
        vkFreeMemory(device, weightStagingMem, nullptr);
    }

    // Recreate outputBuffer
    vkDestroyBuffer(device, outputBuffer, nullptr);
    vkFreeMemory(device, outputBufferMemory, nullptr);
    createBuffer(output_size, storage_buffer_usage, memory_properties, outputBuffer, outputBufferMemory);

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

    // Update descriptors
    VkDescriptorBufferInfo inputBufferInfo{};
    inputBufferInfo.buffer = inputBuffer;
    inputBufferInfo.offset = 0;
    inputBufferInfo.range = input_size;

    std::vector<VkDescriptorBufferInfo> weightBufferInfos;
    for (size_t i = 0; i < weightBuffers.size(); ++i) {
        uint32_t chunk_vocab = (i == num_chunks - 1) ? (config.num_embeddings % chunk_size) : chunk_size;
        VkDeviceSize chunk_weight_size = chunk_vocab * config.embedding_dim * sizeof(float);
        VkDescriptorBufferInfo info{};
        info.buffer = weightBuffers[i];
        info.offset = 0;
        info.range = chunk_weight_size;
        weightBufferInfos.push_back(info);
    }

    VkDescriptorBufferInfo outputBufferInfo{};
    outputBufferInfo.buffer = outputBuffer;
    outputBufferInfo.offset = 0;
    outputBufferInfo.range = output_size;

    VkDescriptorBufferInfo uniformBufferInfo{};
    uniformBufferInfo.buffer = uniformBuffer;
    uniformBufferInfo.offset = 0;
    uniformBufferInfo.range = sizeof(UniformData);

    std::vector<VkWriteDescriptorSet> descriptorWrites = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &inputBufferInfo, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &weightBufferInfos[0], nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &weightBufferInfos[1], nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &weightBufferInfos[2], nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &weightBufferInfos[3], nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &weightBufferInfos[4], nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &weightBufferInfos[5], nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 7, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &weightBufferInfos[6], nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 8, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &weightBufferInfos[7], nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &weightBufferInfos[8], nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &weightBufferInfos[9], nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 11, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &weightBufferInfos[10], nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 12, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &outputBufferInfo, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 13, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uniformBufferInfo, nullptr}
    };

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);

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

    if (vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        throw std::runtime_error("failed to submit embedding compute command buffer!");
    }
    if (vkQueueWaitIdle(computeQueue) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        throw std::runtime_error("failed to wait for embedding queue idle!");
    }
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);

    // Transfer output
    Tensor raw_output;
    raw_output.shape = {static_cast<uint32_t>(input.size()), config.embedding_dim};
    raw_output.data.resize(input.size() * config.embedding_dim);

    VkBuffer outStagingBuffer;
    VkDeviceMemory outStagingBufferMemory;
    createBuffer(output_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, outStagingBuffer, outStagingBufferMemory);
    copyBuffer(outputBuffer, outStagingBuffer, output_size);
    void* outData;
    vkMapMemory(device, outStagingBufferMemory, 0, output_size, 0, &outData);
    memcpy(raw_output.data.data(), outData, output_size);
    vkUnmapMemory(device, outStagingBufferMemory);
    vkDestroyBuffer(device, outStagingBuffer, nullptr);
    vkFreeMemory(device, outStagingBufferMemory, nullptr);

    // Average over sequence dimension to match HRM batch input
    // Input is flattened [batch_size * seq_len], output is [batch_size, embedding_dim]
    uint32_t seq_len = config.seq_len;
    if (input.size() % seq_len != 0) {
        throw std::runtime_error("Input size must be divisible by sequence length");
    }
    uint32_t batch_size = input.size() / seq_len;
    uint32_t embedding_dim = static_cast<uint32_t>(config.embedding_dim);
    Tensor output;
    output.data.resize(batch_size * embedding_dim, 0.0f);
    output.shape = {batch_size, embedding_dim};

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < embedding_dim; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < seq_len; ++k) {
                sum += raw_output.data[(i * seq_len + k) * embedding_dim + j];
            }
            output.data[i * embedding_dim + j] = sum / seq_len;
        }
    }

    return output;
}

std::unordered_map<std::string, Tensor> EmbeddingVulkan::get_parameters() {
    std::unordered_map<std::string, Tensor> params;
    if (device == VK_NULL_HANDLE) {
        // No GPU-backed weights accessible in CPU fallback
        return params;
    }

    // Read all chunked weight buffers and concatenate
    Tensor weights;
    weights.shape = {config.num_embeddings, config.embedding_dim};
    weights.data.resize(static_cast<size_t>(config.num_embeddings) * config.embedding_dim);

    uint32_t offset_vocab = 0;
    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        uint32_t chunk_vocab = (chunk == num_chunks - 1) ? (config.num_embeddings % chunk_size) : chunk_size;
        if (chunk_vocab == 0) chunk_vocab = chunk_size;
        VkDeviceSize chunk_weight_size = static_cast<VkDeviceSize>(chunk_vocab) * config.embedding_dim * sizeof(float);

        VkBuffer stagingBuf;
        VkDeviceMemory stagingMem;
        createBuffer(chunk_weight_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuf, stagingMem);

        copyBuffer(weightBuffers[chunk], stagingBuf, chunk_weight_size);

        void* mapped;
        vkMapMemory(device, stagingMem, 0, chunk_weight_size, 0, &mapped);
        float* src = static_cast<float*>(mapped);
        for (uint32_t i = 0; i < chunk_vocab; ++i) {
            for (uint32_t j = 0; j < config.embedding_dim; ++j) {
                uint32_t global_idx = (offset_vocab + i) * config.embedding_dim + j;
                uint32_t local_idx = i * config.embedding_dim + j;
                weights.data[global_idx] = src[local_idx];
            }
        }
        vkUnmapMemory(device, stagingMem);
        vkDestroyBuffer(device, stagingBuf, nullptr);
        vkFreeMemory(device, stagingMem, nullptr);
        offset_vocab += chunk_vocab;
    }

    params["weight"] = std::move(weights);
    return params;
}

void EmbeddingVulkan::set_parameters(const std::unordered_map<std::string, Tensor>& params) {
    if (device == VK_NULL_HANDLE) return;
    auto it = params.find("weight");
    if (it == params.end()) return;

    const Tensor& weights = it->second;
    if (weights.data.empty()) return;

    // Upload chunked weights to GPU buffers
    uint32_t offset_vocab = 0;
    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        uint32_t chunk_vocab = (chunk == num_chunks - 1) ? (config.num_embeddings % chunk_size) : chunk_size;
        if (chunk_vocab == 0) chunk_vocab = chunk_size;
        VkDeviceSize chunk_weight_size = static_cast<VkDeviceSize>(chunk_vocab) * config.embedding_dim * sizeof(float);

        VkBuffer stagingBuf;
        VkDeviceMemory stagingMem;
        createBuffer(chunk_weight_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuf, stagingMem);

        void* mapped;
        vkMapMemory(device, stagingMem, 0, chunk_weight_size, 0, &mapped);
        float* dst = static_cast<float*>(mapped);
        for (uint32_t i = 0; i < chunk_vocab; ++i) {
            for (uint32_t j = 0; j < config.embedding_dim; ++j) {
                uint32_t global_idx = (offset_vocab + i) * config.embedding_dim + j;
                uint32_t local_idx = i * config.embedding_dim + j;
                dst[local_idx] = weights.data[global_idx];
            }
        }
        vkUnmapMemory(device, stagingMem);

        copyBuffer(stagingBuf, weightBuffers[chunk], chunk_weight_size);
        vkDestroyBuffer(device, stagingBuf, nullptr);
        vkFreeMemory(device, stagingMem, nullptr);
        offset_vocab += chunk_vocab;
    }
}

void EmbeddingVulkan::init_vulkan_objects() {
    createComputePipeline();

    // Create buffers with sufficient sizes
    VkDeviceSize buffer_size = 1024 * 1024; // 1MB, sufficient for initial operations
    VkBufferUsageFlags storage_buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags memory_properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    createBuffer(buffer_size, storage_buffer_usage, memory_properties, inputBuffer, inputBufferMemory);
    for (uint32_t i = 0; i < num_chunks; ++i) {
        VkBuffer buffer;
        VkDeviceMemory memory;
        createBuffer(buffer_size, storage_buffer_usage, memory_properties, buffer, memory);
        weightBuffers.push_back(buffer);
        weightBufferMemories.push_back(memory);
    }
    createBuffer(buffer_size, storage_buffer_usage, memory_properties, outputBuffer, outputBufferMemory);
    createBuffer(sizeof(uint32_t) * 3, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffer, uniformBufferMemory);
    
    // Descriptor pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 13}, // input, 11 weights, output
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
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // weight0
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // weight1
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // weight2
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // weight3
        {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // weight4
        {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // weight5
        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // weight6
        {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // weight7
        {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // weight8
        {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // weight9
        {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // weight10
        {12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // output
        {13, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}  // params
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

std::string EmbeddingVulkan::getShaderPath(const std::string& shaderName) {
    // Try multiple possible locations for shader files

    // 1. Relative to executable (build directory)
    std::string exePath = "shaders/" + shaderName;
    if (fs::exists(exePath)) {
        return exePath;
    }

    // 2. Relative to current working directory
    std::string cwdPath = "./shaders/" + shaderName;
    if (fs::exists(cwdPath)) {
        return cwdPath;
    }

    // 3. Check common installation paths
    std::vector<std::string> searchPaths = {
        "/usr/local/share/nyx/shaders/",
        "/usr/share/nyx/shaders/",
        "/opt/nyx/shaders/"
    };

    for (const auto& basePath : searchPaths) {
        std::string fullPath = basePath + shaderName;
        if (fs::exists(fullPath)) {
            return fullPath;
        }
    }

    // 4. Environment variable override
    if (const char* envPath = std::getenv("NYX_SHADER_PATH")) {
        std::string envFullPath = std::string(envPath) + "/" + shaderName;
        if (fs::exists(envFullPath)) {
            return envFullPath;
        }
    }

    // Fallback to relative path (may fail but provides clear error)
    std::cout << "Warning: Shader file not found in standard locations, trying: " << exePath << std::endl;
    return exePath;
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
    auto shaderCode = readFile(getShaderPath("embedding.spv"));

    VkShaderModuleCreateInfo shaderModuleCI{};
    shaderModuleCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCI.codeSize = shaderCode.size();
    shaderModuleCI.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());

    VkShaderModule computeShaderModule;
    vkCreateShaderModule(device, &shaderModuleCI, nullptr, &computeShaderModule);

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{};
    descriptorSetLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // input
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // weight0
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // weight1
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // weight2
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // weight3
        {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // weight4
        {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // weight5
        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // weight6
        {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // weight7
        {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // weight8
        {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // weight9
        {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // weight10
        {12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // output
        {13, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}  // params
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

    VkResult bufferResult = vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);
    if (bufferResult != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan buffer: " +
                                VulkanCompatibility::getVulkanResultString(bufferResult));
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    VkResult allocResult = vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory);
    if (allocResult != VK_SUCCESS) {
        vkDestroyBuffer(device, buffer, nullptr);
        throw std::runtime_error("Failed to allocate Vulkan buffer memory: " +
                                VulkanCompatibility::getVulkanResultString(allocResult));
    }

    VkResult bindResult = vkBindBufferMemory(device, buffer, bufferMemory, 0);
    if (bindResult != VK_SUCCESS) {
        vkDestroyBuffer(device, buffer, nullptr);
        vkFreeMemory(device, bufferMemory, nullptr);
        throw std::runtime_error("Failed to bind Vulkan buffer memory: " +
                                VulkanCompatibility::getVulkanResultString(bindResult));
    }
}

void EmbeddingVulkan::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    // Check if device is still valid before operations
    if (device == VK_NULL_HANDLE || computeQueue == VK_NULL_HANDLE) {
        return; // Device already destroyed, skip copy
    }
    
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

    if (vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        throw std::runtime_error("failed to submit embedding copy command buffer!");
    }
    if (vkQueueWaitIdle(computeQueue) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        throw std::runtime_error("failed to wait for embedding copy queue idle!");
    }

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

std::vector<Tensor> EmbeddingVulkan::backward(const std::vector<uint32_t>& input, const Tensor& output_grad) {
    // Compute gradients for embedding matrix
    // For each position, accumulate output_grad into the embedding vector for input[token_id]

    Tensor embedding_grad;
    embedding_grad.shape = {config.num_embeddings, config.embedding_dim};
    embedding_grad.data.resize(config.num_embeddings * config.embedding_dim, 0.0f);

    // For each sequence position, accumulate gradients
    for (size_t seq_pos = 0; seq_pos < input.size(); ++seq_pos) {
        uint32_t token_id = input[seq_pos];
        if (token_id < config.num_embeddings) {
            // Accumulate output gradients into embedding gradients
            for (size_t dim = 0; dim < config.embedding_dim; ++dim) {
                size_t output_idx = seq_pos * config.embedding_dim + dim;
                size_t embedding_idx = token_id * config.embedding_dim + dim;
                embedding_grad.data[embedding_idx] += output_grad.data[output_idx];
            }
        }
    }

    return {embedding_grad};
}