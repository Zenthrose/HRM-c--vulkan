#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "attention.hpp"

// VK_CHECK macro for Vulkan error handling
#define VK_CHECK(x) if ((x) != VK_SUCCESS) throw std::runtime_error("Vulkan operation failed with error: " + std::to_string(x));

AttentionVulkan::AttentionVulkan(const AttentionConfig& config, VkPhysicalDevice physicalDevice, VkDevice device, VkQueue computeQueue, uint32_t computeQueueFamilyIndex, VkCommandPool commandPool)
    : config(config), physicalDevice(physicalDevice), device(device), computeQueue(computeQueue), computeQueueFamilyIndex(computeQueueFamilyIndex), commandPool(commandPool),
      pipeline(VK_NULL_HANDLE), pipelineLayout(VK_NULL_HANDLE), descriptorSetLayout(VK_NULL_HANDLE), descriptorPool(VK_NULL_HANDLE) {
    std::cout << "Initializing AttentionVulkan layer..." << std::endl;
    init_vulkan_objects();
}

AttentionVulkan::~AttentionVulkan() {
    std::cout << "Destroying AttentionVulkan layer..." << std::endl;

    // Check if device is still valid before cleanup
    if (device == VK_NULL_HANDLE) {
        return; // Device already destroyed, skip cleanup
    }

    // Clean up staging buffers first
    for (auto& staging : stagingBuffers) {
        if (staging.first != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, staging.first, nullptr);
        }
        if (staging.second != VK_NULL_HANDLE) {
            vkFreeMemory(device, staging.second, nullptr);
        }
    }
    stagingBuffers.clear();



    // Then destroy persistent buffers
    vkDestroyBuffer(device, queryBuffer, nullptr);
    vkFreeMemory(device, queryBufferMemory, nullptr);
    vkDestroyBuffer(device, keyBuffer, nullptr);
    vkFreeMemory(device, keyBufferMemory, nullptr);
    vkDestroyBuffer(device, valueBuffer, nullptr);
    vkFreeMemory(device, valueBufferMemory, nullptr);
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

Tensor AttentionVulkan::forward(const Tensor& hidden_states, const CosSin& cos_sin) {

    // 1. Map uniform buffer and copy config data
    void* uniformData;
    vkMapMemory(device, uniformBufferMemory, 0, sizeof(AttentionConfig), 0, &uniformData);
    memcpy(uniformData, &config, sizeof(AttentionConfig));
    vkUnmapMemory(device, uniformBufferMemory);

    // Calculate actual tensor sizes based on input
    VkDeviceSize input_size = hidden_states.data.size() * sizeof(float);

    // Validate input shape matches config
    if (hidden_states.shape.size() < 3 ||
        hidden_states.shape[0] != config.batch_size ||
        hidden_states.shape[1] != config.seq_len) {
        throw std::runtime_error("Input tensor shape does not match attention config");
    }

    uint32_t hidden_size = hidden_states.shape[2];
    if (hidden_size != config.num_heads * config.head_dim) {
        throw std::runtime_error("Hidden size does not match num_heads * head_dim");
    }

    uint32_t total_elements = config.batch_size * config.seq_len * config.head_dim;

    // For simplified attention, use same data for Q, K, V (identity attention)
    VkDeviceSize q_size = total_elements * sizeof(float);
    VkDeviceSize k_size = total_elements * sizeof(float);
    VkDeviceSize v_size = total_elements * sizeof(float);
    VkDeviceSize out_size = input_size;

    // Transfer Q (use input tensor directly for simplified attention)
    VkBuffer qStagingBuffer;
    VkDeviceMemory qStagingMemory;
    createBuffer(q_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, qStagingBuffer, qStagingMemory);
    stagingBuffers.push_back({qStagingBuffer, qStagingMemory});
    void* qData;
    vkMapMemory(device, qStagingMemory, 0, q_size, 0, &qData);
    memcpy(qData, hidden_states.data.data(), q_size);
    vkUnmapMemory(device, qStagingMemory);
    copyBuffer(qStagingBuffer, queryBuffer, q_size);

    // Transfer K
    VkBuffer kStagingBuffer;
    VkDeviceMemory kStagingMemory;
    createBuffer(k_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, kStagingBuffer, kStagingMemory);
    stagingBuffers.push_back({kStagingBuffer, kStagingMemory});
    void* kData;
    vkMapMemory(device, kStagingMemory, 0, k_size, 0, &kData);
    memcpy(kData, hidden_states.data.data() + q_size/sizeof(float), k_size);
    vkUnmapMemory(device, kStagingMemory);
    copyBuffer(kStagingBuffer, keyBuffer, k_size);

    // Transfer V
    VkBuffer vStagingBuffer;
    VkDeviceMemory vStagingMemory;
    createBuffer(v_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vStagingBuffer, vStagingMemory);
    stagingBuffers.push_back({vStagingBuffer, vStagingMemory});
    void* vData;
    vkMapMemory(device, vStagingMemory, 0, v_size, 0, &vData);
    memcpy(vData, hidden_states.data.data() + (q_size + k_size)/sizeof(float), v_size);
    vkUnmapMemory(device, vStagingMemory);
    copyBuffer(vStagingBuffer, valueBuffer, v_size);

    // 3. Record and Submit Command Buffer
    VkCommandBufferAllocateInfo cmdBufAllocInfo{};
    cmdBufAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdBufAllocInfo.commandPool = commandPool;
    cmdBufAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufAllocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    VK_CHECK(vkAllocateCommandBuffers(device, &cmdBufAllocInfo, &commandBuffer));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    // Dispatch parameters: one workgroup per (batch, head, seq_i) triple
    uint32_t groupCountX = config.batch_size * config.num_heads * config.seq_len;

    // Check device limits
    VkPhysicalDeviceProperties deviceProps;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProps);
    if (groupCountX > deviceProps.limits.maxComputeWorkGroupCount[0]) {
        throw std::runtime_error("Attention dispatch size exceeds device limits");
    }

    vkCmdDispatch(commandBuffer, groupCountX, 1, 1);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    // Add fence for synchronization
    VkFence fence;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &fence));
    }

    if (vkQueueSubmit(computeQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
        vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        throw std::runtime_error("failed to submit compute command buffer!");
    }
    
    // Wait for completion with timeout
    VkResult waitResult = vkWaitForFences(device, 1, &fence, VK_TRUE, 10000000000ULL); // 10 second timeout
    VK_CHECK(waitResult);
    
    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);

    std::cout << "Compute shader dispatched and finished." << std::endl;

    // 4. Transfer Output Data
    Tensor output;
    output.shape = {config.batch_size, config.seq_len, config.num_heads * config.head_dim};
    output.data.resize(out_size / sizeof(float));

    VkBuffer outStagingBuffer;
    VkDeviceMemory outStagingBufferMemory;
    createBuffer(out_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, outStagingBuffer, outStagingBufferMemory);
    stagingBuffers.push_back({outStagingBuffer, outStagingBufferMemory});
    copyBuffer(outputBuffer, outStagingBuffer, out_size);
    void* outData;
    vkMapMemory(device, outStagingBufferMemory, 0, out_size, 0, &outData);
    memcpy(output.data.data(), outData, out_size);
    vkUnmapMemory(device, outStagingBufferMemory);

    return output;
}

std::pair<Tensor, std::vector<Tensor>> AttentionVulkan::backward(const Tensor& hidden_states, const Tensor& output_grad) {
    // Implement CPU backward for the simplified attention used by tests.
    // Assumptions:
    // - hidden_states.data is: [Q_flat..., K_flat..., V_flat...] where
    //   Q: (batch, seq, num_heads, head_dim)
    //   K: (batch, seq, num_key_value_heads, head_dim)
    //   V: (batch, seq, num_key_value_heads, head_dim)
    // - output_grad has same layout as the attention output: (batch, seq, num_heads, head_dim)

    const uint32_t B = config.batch_size;
    const uint32_t S = config.seq_len;
    const uint32_t H = config.num_heads;
    const uint32_t Hk = config.num_key_value_heads;
    const uint32_t D = config.head_dim;

    size_t q_elems = static_cast<size_t>(B) * S * H * D;
    size_t k_elems = static_cast<size_t>(B) * S * Hk * D;
    size_t v_elems = k_elems;

    if (hidden_states.data.size() < q_elems + k_elems + v_elems) {
        throw std::runtime_error("hidden_states does not contain enough data for Q,K,V");
    }

    // Slices into input
    const float* q_ptr = hidden_states.data.data();
    const float* k_ptr = q_ptr + q_elems;
    const float* v_ptr = k_ptr + k_elems;

    // Output gradient pointer and shape checks
    size_t out_elems = static_cast<size_t>(B) * S * H * D;
    if (output_grad.data.size() < out_elems) {
        throw std::runtime_error("output_grad has insufficient size");
    }
    const float* outg_ptr = output_grad.data.data();

    // Prepare gradients for Q,K,V
    std::vector<float> dq(q_elems, 0.0f);
    std::vector<float> dk(k_elems, 0.0f);
    std::vector<float> dv(v_elems, 0.0f);

    auto idx4 = [](uint32_t b, uint32_t s, uint32_t h, uint32_t d, uint32_t B, uint32_t S, uint32_t H, uint32_t D) {
        return (((size_t)b * S + s) * H + h) * D + d;
    };

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // For each batch and head, compute attention and propagate gradients
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t h = 0; h < H; ++h) {
            uint32_t hk = h % Hk; // key/value head index

            // Compute scores: S x S matrix
            std::vector<float> scores(S * S);
            for (uint32_t qi = 0; qi < S; ++qi) {
                for (uint32_t ki = 0; ki < S; ++ki) {
                    float s_val = 0.0f;
                    for (uint32_t d = 0; d < D; ++d) {
                        size_t q_index = idx4(b, qi, h, d, B, S, H, D);
                        size_t k_index = idx4(b, ki, hk, d, B, S, Hk, D);
                        float qv = q_ptr[q_index];
                        float kv = k_ptr[k_index];
                        s_val += qv * kv;
                    }
                    scores[qi * S + ki] = s_val * scale;
                }
            }

            // Apply causal mask if needed and compute softmax per query
            std::vector<float> attn(S * S);
            for (uint32_t qi = 0; qi < S; ++qi) {
                // find max for numerical stability
                float m = -std::numeric_limits<float>::infinity();
                for (uint32_t ki = 0; ki < S; ++ki) {
                    if (config.causal && ki > qi) continue; // masked
                    m = std::max(m, scores[qi * S + ki]);
                }
                float sum = 0.0f;
                for (uint32_t ki = 0; ki < S; ++ki) {
                    if (config.causal && ki > qi) {
                        attn[qi * S + ki] = 0.0f;
                        continue;
                    }
                    float e = std::exp(scores[qi * S + ki] - m);
                    attn[qi * S + ki] = e;
                    sum += e;
                }
                for (uint32_t ki = 0; ki < S; ++ki) {
                    if (config.causal && ki > qi) continue;
                    attn[qi * S + ki] /= sum;
                }
            }

            // Compute gradients
            // For dv: dv[k] += sum_q attn[q,k] * dout[q]
            for (uint32_t ki = 0; ki < S; ++ki) {
                for (uint32_t d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (uint32_t qi = 0; qi < S; ++qi) {
                        size_t outg_index = idx4(b, qi, h, d, B, S, H, D);
                        float g = outg_ptr[outg_index];
                        float a = attn[qi * S + ki];
                        acc += a * g;
                    }
                    size_t v_index = idx4(b, ki, hk, d, B, S, Hk, D);
                    dv[v_index] += acc;
                }
            }

            // dAttn = dout * V^T  (for each query q and key k)
            std::vector<float> dAttn(S * S, 0.0f);
            for (uint32_t qi = 0; qi < S; ++qi) {
                for (uint32_t ki = 0; ki < S; ++ki) {
                    float acc = 0.0f;
                    for (uint32_t d = 0; d < D; ++d) {
                        size_t outg_index = idx4(b, qi, h, d, B, S, H, D);
                        size_t v_index = idx4(b, ki, hk, d, B, S, Hk, D);
                        float g = outg_ptr[outg_index];
                        float vval = v_ptr[v_index];
                        acc += g * vval;
                    }
                    dAttn[qi * S + ki] = acc;
                }
            }

            // Backprop through softmax: for each query qi, compute dScore
            std::vector<float> dScore(S * S, 0.0f);
            for (uint32_t qi = 0; qi < S; ++qi) {
                // compute dot = sum_k attn_k * dAttn_k
                float dot = 0.0f;
                for (uint32_t k2 = 0; k2 < S; ++k2) {
                    dot += attn[qi * S + k2] * dAttn[qi * S + k2];
                }
                for (uint32_t ki = 0; ki < S; ++ki) {
                    if (config.causal && ki > qi) continue;
                    float ds = attn[qi * S + ki] * (dAttn[qi * S + ki] - dot);
                    dScore[qi * S + ki] = ds;
                }
            }

            // Multiply by scaling factor (scores = Q*K^T * scale)
            for (size_t i = 0; i < dScore.size(); ++i) dScore[i] *= scale;

            // dQ += dScore * K
            for (uint32_t qi = 0; qi < S; ++qi) {
                for (uint32_t d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (uint32_t ki = 0; ki < S; ++ki) {
                        size_t k_index = idx4(b, ki, hk, d, B, S, Hk, D);
                        float kval = k_ptr[k_index];
                        acc += dScore[qi * S + ki] * kval;
                    }
                    size_t q_index = idx4(b, qi, h, d, B, S, H, D);
                    dq[q_index] += acc;
                }
            }

            // dK += dScore^T * Q  (for each key position)
            for (uint32_t ki = 0; ki < S; ++ki) {
                for (uint32_t d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (uint32_t qi = 0; qi < S; ++qi) {
                        size_t q_index = idx4(b, qi, h, d, B, S, H, D);
                        float qval = q_ptr[q_index];
                        acc += dScore[qi * S + ki] * qval;
                    }
                    size_t k_index = idx4(b, ki, hk, d, B, S, Hk, D);
                    dk[k_index] += acc;
                }
            }
        }
    }

    // Pack input_grad in same layout as hidden_states: Q,K,V concatenated
    Tensor input_grad;
    input_grad.shape = hidden_states.shape;
    input_grad.data.resize(q_elems + k_elems + v_elems);
    std::copy(dq.begin(), dq.end(), input_grad.data.begin());
    std::copy(dk.begin(), dk.end(), input_grad.data.begin() + q_elems);
    std::copy(dv.begin(), dv.end(), input_grad.data.begin() + q_elems + k_elems);

    return cpu_attention_backward(hidden_states, output_grad, config);
}

// CPU reference backward implementation (callable without Vulkan objects)
std::pair<Tensor, std::vector<Tensor>> cpu_attention_backward(const Tensor& hidden_states, const Tensor& output_grad, const AttentionConfig& cfg) {
    // We reuse the implementation above but operating on the provided config
    // Duplicate of AttentionVulkan::backward's logic but using cfg.
    const uint32_t B = cfg.batch_size;
    const uint32_t S = cfg.seq_len;
    const uint32_t H = cfg.num_heads;
    const uint32_t Hk = cfg.num_key_value_heads;
    const uint32_t D = cfg.head_dim;

    size_t q_elems = static_cast<size_t>(B) * S * H * D;
    size_t k_elems = static_cast<size_t>(B) * S * Hk * D;
    size_t v_elems = k_elems;

    if (hidden_states.data.size() < q_elems + k_elems + v_elems) {
        throw std::runtime_error("hidden_states does not contain enough data for Q,K,V");
    }

    const float* q_ptr = hidden_states.data.data();
    const float* k_ptr = q_ptr + q_elems;
    const float* v_ptr = k_ptr + k_elems;

    size_t out_elems = static_cast<size_t>(B) * S * H * D;
    if (output_grad.data.size() < out_elems) {
        throw std::runtime_error("output_grad has insufficient size");
    }
    const float* outg_ptr = output_grad.data.data();

    std::vector<float> dq(q_elems, 0.0f);
    std::vector<float> dk(k_elems, 0.0f);
    std::vector<float> dv(v_elems, 0.0f);

    auto idx4 = [](uint32_t b, uint32_t s, uint32_t h, uint32_t d, uint32_t B, uint32_t S, uint32_t H, uint32_t D) {
        return (((size_t)b * S + s) * H + h) * D + d;
    };

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t h = 0; h < H; ++h) {
            uint32_t hk = h % Hk;

            std::vector<float> scores(S * S);
            for (uint32_t qi = 0; qi < S; ++qi) {
                for (uint32_t ki = 0; ki < S; ++ki) {
                    float s_val = 0.0f;
                    for (uint32_t d = 0; d < D; ++d) {
                        size_t q_index = idx4(b, qi, h, d, B, S, H, D);
                        size_t k_index = idx4(b, ki, hk, d, B, S, Hk, D);
                        float qv = q_ptr[q_index];
                        float kv = k_ptr[k_index];
                        s_val += qv * kv;
                    }
                    scores[qi * S + ki] = s_val * scale;
                }
            }

            std::vector<float> attn(S * S);
            for (uint32_t qi = 0; qi < S; ++qi) {
                float m = -std::numeric_limits<float>::infinity();
                for (uint32_t ki = 0; ki < S; ++ki) {
                    if (cfg.causal && ki > qi) continue;
                    m = std::max(m, scores[qi * S + ki]);
                }
                float sum = 0.0f;
                for (uint32_t ki = 0; ki < S; ++ki) {
                    if (cfg.causal && ki > qi) {
                        attn[qi * S + ki] = 0.0f;
                        continue;
                    }
                    float e = std::exp(scores[qi * S + ki] - m);
                    attn[qi * S + ki] = e;
                    sum += e;
                }
                for (uint32_t ki = 0; ki < S; ++ki) {
                    if (cfg.causal && ki > qi) continue;
                    attn[qi * S + ki] /= sum;
                }
            }

            for (uint32_t ki = 0; ki < S; ++ki) {
                for (uint32_t d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (uint32_t qi = 0; qi < S; ++qi) {
                        size_t outg_index = idx4(b, qi, h, d, B, S, H, D);
                        float g = outg_ptr[outg_index];
                        float a = attn[qi * S + ki];
                        acc += a * g;
                    }
                    size_t v_index = idx4(b, ki, hk, d, B, S, Hk, D);
                    dv[v_index] += acc;
                }
            }

            std::vector<float> dAttn(S * S, 0.0f);
            for (uint32_t qi = 0; qi < S; ++qi) {
                for (uint32_t ki = 0; ki < S; ++ki) {
                    float acc = 0.0f;
                    for (uint32_t d = 0; d < D; ++d) {
                        size_t outg_index = idx4(b, qi, h, d, B, S, H, D);
                        size_t v_index = idx4(b, ki, hk, d, B, S, Hk, D);
                        float g = outg_ptr[outg_index];
                        float vval = v_ptr[v_index];
                        acc += g * vval;
                    }
                    dAttn[qi * S + ki] = acc;
                }
            }

            std::vector<float> dScore(S * S, 0.0f);
            for (uint32_t qi = 0; qi < S; ++qi) {
                float dot = 0.0f;
                for (uint32_t k2 = 0; k2 < S; ++k2) {
                    dot += attn[qi * S + k2] * dAttn[qi * S + k2];
                }
                for (uint32_t ki = 0; ki < S; ++ki) {
                    if (cfg.causal && ki > qi) continue;
                    float ds = attn[qi * S + ki] * (dAttn[qi * S + ki] - dot);
                    dScore[qi * S + ki] = ds;
                }
            }

            for (size_t i = 0; i < dScore.size(); ++i) dScore[i] *= scale;

            for (uint32_t qi = 0; qi < S; ++qi) {
                for (uint32_t d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (uint32_t ki = 0; ki < S; ++ki) {
                        size_t k_index = idx4(b, ki, hk, d, B, S, Hk, D);
                        float kval = k_ptr[k_index];
                        acc += dScore[qi * S + ki] * kval;
                    }
                    size_t q_index = idx4(b, qi, h, d, B, S, H, D);
                    dq[q_index] += acc;
                }
            }

            for (uint32_t ki = 0; ki < S; ++ki) {
                for (uint32_t d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (uint32_t qi = 0; qi < S; ++qi) {
                        size_t q_index = idx4(b, qi, h, d, B, S, H, D);
                        float qval = q_ptr[q_index];
                        acc += dScore[qi * S + ki] * qval;
                    }
                    size_t k_index = idx4(b, ki, hk, d, B, S, Hk, D);
                    dk[k_index] += acc;
                }
            }
        }
    }

    Tensor input_grad;
    input_grad.shape = hidden_states.shape;
    input_grad.data.resize(q_elems + k_elems + v_elems);
    std::copy(dq.begin(), dq.end(), input_grad.data.begin());
    std::copy(dk.begin(), dk.end(), input_grad.data.begin() + q_elems);
    std::copy(dv.begin(), dv.end(), input_grad.data.begin() + q_elems + k_elems);

    std::vector<Tensor> param_grads;
    return {input_grad, param_grads};
}

void AttentionVulkan::init_vulkan_objects() {
    createComputePipeline();

    VkDeviceSize buffer_size = config.batch_size * config.seq_len * config.num_heads * config.head_dim * sizeof(float);
    VkBufferUsageFlags storage_buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags memory_properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    createBuffer(buffer_size, storage_buffer_usage, memory_properties, queryBuffer, queryBufferMemory);
    createBuffer(buffer_size, storage_buffer_usage, memory_properties, keyBuffer, keyBufferMemory);
    createBuffer(buffer_size, storage_buffer_usage, memory_properties, valueBuffer, valueBufferMemory);
    createBuffer(buffer_size, storage_buffer_usage, memory_properties, outputBuffer, outputBufferMemory);
    createBuffer(sizeof(AttentionConfig), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffer, uniformBufferMemory);

    // Descriptor pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}, // Q, K, V, Output
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
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Q
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // K
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // V
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Output
        {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}  // Config
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

    // Update descriptor set
    VkDescriptorBufferInfo q_info = {queryBuffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo k_info = {keyBuffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo v_info = {valueBuffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo out_info = {outputBuffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo uniform_info = {uniformBuffer, 0, VK_WHOLE_SIZE};

    std::vector<VkWriteDescriptorSet> descriptorWrites(5);
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &q_info;

    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = descriptorSet;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pBufferInfo = &k_info;

    descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[2].dstSet = descriptorSet;
    descriptorWrites[2].dstBinding = 2;
    descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[2].descriptorCount = 1;
    descriptorWrites[2].pBufferInfo = &v_info;

    descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[3].dstSet = descriptorSet;
    descriptorWrites[3].dstBinding = 3;
    descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[3].descriptorCount = 1;
    descriptorWrites[3].pBufferInfo = &out_info;

    descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[4].dstSet = descriptorSet;
    descriptorWrites[4].dstBinding = 4;
    descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[4].descriptorCount = 1;
    descriptorWrites[4].pBufferInfo = &uniform_info;

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
}

std::vector<char> AttentionVulkan::readFile(const std::string& filename) {
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

void AttentionVulkan::createComputePipeline() {
    auto shaderCode = readFile("shaders/attention.spv");

    VkShaderModuleCreateInfo shaderModuleCI{};
    shaderModuleCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCI.codeSize = shaderCode.size();
    shaderModuleCI.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());

    VkShaderModule computeShaderModule;
    vkCreateShaderModule(device, &shaderModuleCI, nullptr, &computeShaderModule);

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{};
    descriptorSetLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
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

uint32_t AttentionVulkan::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void AttentionVulkan::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory);
    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

void AttentionVulkan::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
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

    VK_CHECK(vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(computeQueue));

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}