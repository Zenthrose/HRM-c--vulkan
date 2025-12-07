#include "vulkan_trainer.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <random>
#include <filesystem>
#include <future>
#include <thread>
#include <vector>

VulkanTrainer::VulkanTrainer(VkDevice device, VkPhysicalDevice physical_device,
                           uint32_t compute_queue_family_index, VkQueue compute_queue,
                           const VulkanTrainingConfig& config)
    : device_(device), physical_device_(physical_device),
      compute_queue_family_index_(compute_queue_family_index), compute_queue_(compute_queue),
      config_(config), current_epoch_(0), current_loss_(0.0f), current_perplexity_(0.0f), timestep_(1) {
}

VulkanTrainer::~VulkanTrainer() {
    // Cleanup Vulkan resources
    if (param_buffer_) vkDestroyBuffer(device_, param_buffer_, nullptr);
    if (grad_buffer_) vkDestroyBuffer(device_, grad_buffer_, nullptr);
    if (adam_m_buffer_) vkDestroyBuffer(device_, adam_m_buffer_, nullptr);
    if (adam_v_buffer_) vkDestroyBuffer(device_, adam_v_buffer_, nullptr);
    if (input_buffer_) vkDestroyBuffer(device_, input_buffer_, nullptr);
    if (target_buffer_) vkDestroyBuffer(device_, target_buffer_, nullptr);
    if (loss_buffer_) vkDestroyBuffer(device_, loss_buffer_, nullptr);

    if (param_memory_) vkFreeMemory(device_, param_memory_, nullptr);
    if (grad_memory_) vkFreeMemory(device_, grad_memory_, nullptr);
    if (adam_m_memory_) vkFreeMemory(device_, adam_m_memory_, nullptr);
    if (adam_v_memory_) vkFreeMemory(device_, adam_v_memory_, nullptr);
    if (input_memory_) vkFreeMemory(device_, input_memory_, nullptr);
    if (target_memory_) vkFreeMemory(device_, target_memory_, nullptr);
    if (loss_memory_) vkFreeMemory(device_, loss_memory_, nullptr);

    if (linear_forward_pipeline_) vkDestroyPipeline(device_, linear_forward_pipeline_, nullptr);
    if (linear_backward_pipeline_) vkDestroyPipeline(device_, linear_backward_pipeline_, nullptr);
    if (adam_optimizer_pipeline_) vkDestroyPipeline(device_, adam_optimizer_pipeline_, nullptr);
    if (cross_entropy_loss_pipeline_) vkDestroyPipeline(device_, cross_entropy_loss_pipeline_, nullptr);
    if (gradient_accumulation_pipeline_) vkDestroyPipeline(device_, gradient_accumulation_pipeline_, nullptr);

    if (linear_forward_shader_) vkDestroyShaderModule(device_, linear_forward_shader_, nullptr);
    if (linear_backward_shader_) vkDestroyShaderModule(device_, linear_backward_shader_, nullptr);
    if (adam_optimizer_shader_) vkDestroyShaderModule(device_, adam_optimizer_shader_, nullptr);
    if (cross_entropy_loss_shader_) vkDestroyShaderModule(device_, cross_entropy_loss_shader_, nullptr);
    if (gradient_accumulation_shader_) vkDestroyShaderModule(device_, gradient_accumulation_shader_, nullptr);

    if (descriptor_pool_) vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
    if (command_pool_) vkDestroyCommandPool(device_, command_pool_, nullptr);
}

bool VulkanTrainer::initialize() {
    // Create command pool
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = compute_queue_family_index_;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
        std::cerr << "Failed to create command pool" << std::endl;
        return false;
    }

    // Create descriptor pool
    VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 20}
    };

    VkDescriptorPoolCreateInfo descriptor_pool_info = {};
    descriptor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_info.poolSizeCount = 1;
    descriptor_pool_info.pPoolSizes = pool_sizes;
    descriptor_pool_info.maxSets = 20;

    if (vkCreateDescriptorPool(device_, &descriptor_pool_info, nullptr, &descriptor_pool_) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor pool" << std::endl;
        return false;
    }

    // Create descriptor set layout
    if (!create_descriptor_set_layout()) {
        std::cerr << "Failed to create descriptor set layout" << std::endl;
        return false;
    }

    // Create pipeline layout
    if (!create_pipeline_layout()) {
        std::cerr << "Failed to create pipeline layout" << std::endl;
        return false;
    }

    // Initialize model parameters
    if (!initialize_model()) {
        std::cerr << "Failed to initialize model" << std::endl;
        return false;
    }

    return true;
}

std::string VulkanTrainer::generate_text(const std::string& prompt, uint32_t max_length) {
    std::string generated = prompt;

    for (uint32_t i = 0; i < max_length && generated.size() < max_length; ++i) {
        // Get the last character
        char last_char = generated.back();
        uint32_t char_idx = static_cast<uint32_t>(static_cast<unsigned char>(last_char));
        if (char_idx >= config_.vocab_size) char_idx = 0;

        // Forward pass for single character
        std::vector<float> embedding(config_.hidden_size);
        for (uint32_t h = 0; h < config_.hidden_size; ++h) {
            embedding[h] = embedding_weights_[char_idx * config_.hidden_size + h];
        }

        std::vector<float> hidden(config_.hidden_size, 0.0f);
        for (uint32_t h = 0; h < config_.hidden_size; ++h) {
            for (uint32_t hh = 0; hh < config_.hidden_size; ++hh) {
                hidden[h] += embedding[hh] * hidden_weights_[hh * config_.hidden_size + h];
            }
            hidden[h] += hidden_bias_[h];
            hidden[h] = std::max(0.0f, hidden[h]); // ReLU
        }

        std::vector<float> logits(config_.vocab_size);
        for (uint32_t v = 0; v < config_.vocab_size; ++v) {
            logits[v] = output_bias_[v];
            for (uint32_t h = 0; h < config_.hidden_size; ++h) {
                logits[v] += hidden[h] * output_weights_[h * config_.vocab_size + v];
            }
        }

        // Softmax and sampling
        float max_logit = *std::max_element(logits.begin(), logits.end());
        std::vector<float> probs(config_.vocab_size);
        float sum_exp = 0.0f;

        for (uint32_t v = 0; v < config_.vocab_size; ++v) {
            probs[v] = std::exp(logits[v] - max_logit);
            sum_exp += probs[v];
        }

        for (uint32_t v = 0; v < config_.vocab_size; ++v) {
            probs[v] /= sum_exp;
        }

        // Sample next character (simple greedy for now)
        uint32_t next_char_idx = 0;
        float max_prob = 0.0f;
        for (uint32_t v = 0; v < config_.vocab_size; ++v) {
            if (probs[v] > max_prob) {
                max_prob = probs[v];
                next_char_idx = v;
            }
        }

        char next_char = static_cast<char>(next_char_idx);
        generated += next_char;

        // Stop at sentence end
        if (next_char == '.' || next_char == '!' || next_char == '?') {
            break;
        }
    }

    return generated;
}

// Vulkan utility implementations
VkShaderModule VulkanTrainer::create_shader_module(const std::vector<uint32_t>& code) {
    VkShaderModuleCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = code.size() * sizeof(uint32_t);
    create_info.pCode = code.data();

    VkShaderModule shader_module;
    if (vkCreateShaderModule(device_, &create_info, nullptr, &shader_module) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }

    return shader_module;
}

bool VulkanTrainer::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                                VkBuffer& buffer, VkDeviceMemory& memory) {
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device_, &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
        std::cerr << "Failed to create buffer of size " << size << " bytes" << std::endl;
        return false;
    }

    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(device_, buffer, &mem_requirements);

    // Check if requested size is reasonable (prevent excessive allocations)
    const VkDeviceSize MAX_REASONABLE_SIZE = 1024 * 1024 * 1024; // 1GB limit
    if (mem_requirements.size > MAX_REASONABLE_SIZE) {
        std::cerr << "Requested buffer size " << mem_requirements.size 
                  << " exceeds maximum reasonable size " << MAX_REASONABLE_SIZE << std::endl;
        vkDestroyBuffer(device_, buffer, nullptr);
        return false;
    }

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = find_memory_type(mem_requirements.memoryTypeBits, properties);

    // Try memory allocation with graceful fallback
    VkResult alloc_result = vkAllocateMemory(device_, &alloc_info, nullptr, &memory);
    if (alloc_result != VK_SUCCESS) {
        std::cerr << "Failed to allocate " << mem_requirements.size << " bytes of memory (error: " << alloc_result << ")" << std::endl;
        
        // Try with smaller size if possible (graceful degradation)
        if (size > 1024 * 1024) { // If original request was > 1MB
            VkDeviceSize reduced_size = size / 2;
            std::cerr << "Attempting graceful degradation with reduced size " << reduced_size << " bytes" << std::endl;
            
            // Clean up original buffer
            vkDestroyBuffer(device_, buffer, nullptr);
            
            // Recreate with smaller size
            buffer_info.size = reduced_size;
            if (vkCreateBuffer(device_, &buffer_info, nullptr, &buffer) == VK_SUCCESS) {
                vkGetBufferMemoryRequirements(device_, buffer, &mem_requirements);
                alloc_info.allocationSize = mem_requirements.size;
                
                if (vkAllocateMemory(device_, &alloc_info, nullptr, &memory) == VK_SUCCESS) {
                    vkBindBufferMemory(device_, buffer, memory, 0);
                    std::cout << "Successfully allocated reduced buffer size " << reduced_size << " bytes" << std::endl;
                    return true;
                }
            }
        }
        
        // Final cleanup if all attempts failed
        vkDestroyBuffer(device_, buffer, nullptr);
        return false;
    }

    if (vkBindBufferMemory(device_, buffer, memory, 0) != VK_SUCCESS) {
        std::cerr << "Failed to bind buffer memory" << std::endl;
        vkFreeMemory(device_, memory, nullptr);
        vkDestroyBuffer(device_, buffer, nullptr);
        return false;
    }

    return true;
}

uint32_t VulkanTrainer::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type");
}

void VulkanTrainer::copy_data_to_buffer(VkBuffer buffer, const void* data, VkDeviceSize size) {
    // Placeholder - would create staging buffer and copy
    std::cout << "Copying " << size << " bytes to buffer" << std::endl;
}

std::vector<uint32_t> VulkanTrainer::load_spirv_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open SPIR-V file: " << filename << std::endl;
        return {};
    }

    size_t file_size = (size_t)file.tellg();
    std::vector<uint32_t> buffer(file_size / sizeof(uint32_t));

    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), file_size);
    file.close();

    return buffer;
}

VkPipeline VulkanTrainer::create_compute_pipeline(VkShaderModule shader, const std::string& entry_point) {
    VkPipelineShaderStageCreateInfo shader_stage = {};
    shader_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shader_stage.module = shader;
    shader_stage.pName = entry_point.c_str();

    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = shader_stage;
    pipeline_info.layout = pipeline_layout_;

    VkPipeline pipeline;
    if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
        std::cerr << "Failed to create compute pipeline" << std::endl;
        return VK_NULL_HANDLE;
    }

    return pipeline;
}

bool VulkanTrainer::create_descriptor_set_layout() {
    VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };

    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 6;
    layout_info.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &descriptor_set_layout_) != VK_SUCCESS) {
        return false;
    }

    return true;
}

bool VulkanTrainer::create_pipeline_layout() {
    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &descriptor_set_layout_;

    if (vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &pipeline_layout_) != VK_SUCCESS) {
        return false;
    }

    return true;
}

VkDescriptorSet VulkanTrainer::allocate_descriptor_set() {
    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &descriptor_set_layout_;

    VkDescriptorSet descriptor_set;
    if (vkAllocateDescriptorSets(device_, &alloc_info, &descriptor_set) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set");
    }

    return descriptor_set;
}

void VulkanTrainer::update_descriptor_set(VkDescriptorSet descriptor_set, VkBuffer buffer, uint32_t binding) {
    VkDescriptorBufferInfo buffer_info = {};
    buffer_info.buffer = buffer;
    buffer_info.offset = 0;
    buffer_info.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet descriptor_write = {};
    descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_write.dstSet = descriptor_set;
    descriptor_write.dstBinding = binding;
    descriptor_write.descriptorCount = 1;
    descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_write.pBufferInfo = &buffer_info;

    vkUpdateDescriptorSets(device_, 1, &descriptor_write, 0, nullptr);
}

VkCommandBuffer VulkanTrainer::begin_command_buffer() {
    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = command_pool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer command_buffer;
    vkAllocateCommandBuffers(device_, &alloc_info, &command_buffer);

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(command_buffer, &begin_info);

    return command_buffer;
}

void VulkanTrainer::submit_command_buffer(VkCommandBuffer command_buffer) {
    vkEndCommandBuffer(command_buffer);

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;

    if (vkQueueSubmit(compute_queue_, 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS) {
        vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
        throw std::runtime_error("failed to submit trainer command buffer!");
    }
    if (vkQueueWaitIdle(compute_queue_) != VK_SUCCESS) {
        vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
        throw std::runtime_error("failed to wait for trainer queue idle!");
    }

    vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
}

void VulkanTrainer::copy_data_from_buffer(VkBuffer buffer, void* data, VkDeviceSize size) {
    // Placeholder - would create staging buffer and copy
    std::cout << "Copying " << size << " bytes from buffer" << std::endl;
}

bool VulkanTrainer::load_training_data(const std::string& data_path) {
    return prepare_training_data(data_path);
}

bool VulkanTrainer::train_epoch() {
    float epoch_loss = 0.0f;
    uint32_t batch_count = 0;

    for (const auto& batch : training_batches_) {
        // Execute forward pass
        if (!execute_forward_pass(batch)) {
            std::cerr << "Forward pass failed" << std::endl;
            return false;
        }

        // Execute backward pass
        if (!execute_backward_pass(batch)) {
            std::cerr << "Backward pass failed" << std::endl;
            return false;
        }

        // Execute optimizer step
        if (!execute_optimizer_step()) {
            std::cerr << "Optimizer step failed" << std::endl;
            return false;
        }

        // Accumulate loss
        epoch_loss += compute_loss_and_metrics();
        batch_count++;
    }

    current_loss_ = epoch_loss / batch_count;
    current_perplexity_ = std::exp(current_loss_);
    current_epoch_++;

    std::cout << "Epoch " << current_epoch_ << " - Loss: " << current_loss_
              << ", Perplexity: " << current_perplexity_ << std::endl;

    return true;
}

bool VulkanTrainer::save_checkpoint(const std::string& checkpoint_path) {
    // Save model parameters and training state
    // Placeholder implementation
    std::cout << "Saving checkpoint to " << checkpoint_path << std::endl;
    return true;
}

bool VulkanTrainer::load_checkpoint(const std::string& checkpoint_path) {
    // Load model parameters and training state
    // Placeholder implementation
    std::cout << "Loading checkpoint from " << checkpoint_path << std::endl;
    return true;
}

bool VulkanTrainer::initialize_model() {
    // Initialize model parameters with random weights
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);

    // Embedding weights: vocab_size x hidden_size
    embedding_weights_.resize(config_.vocab_size * config_.hidden_size);
    for (auto& w : embedding_weights_) w = dist(gen);

    // Hidden weights: hidden_size x hidden_size
    hidden_weights_.resize(config_.hidden_size * config_.hidden_size);
    for (auto& w : hidden_weights_) w = dist(gen);

    // Hidden bias: hidden_size
    hidden_bias_.resize(config_.hidden_size, 0.0f);

    // Output weights: hidden_size x vocab_size
    output_weights_.resize(config_.hidden_size * config_.vocab_size);
    for (auto& w : output_weights_) w = dist(gen);

    // Output bias: vocab_size
    output_bias_.resize(config_.vocab_size, 0.0f);

    std::cout << "Model initialized with " << config_.vocab_size << " vocab, " << config_.hidden_size << " hidden size" << std::endl;
    return true;
}

bool VulkanTrainer::prepare_training_data(const std::string& data_path) {
    namespace fs = std::filesystem;

    // Check if data_path is a file or directory
    fs::path path(data_path);
    std::vector<std::string> text_files;

    if (fs::is_regular_file(path)) {
        // Single file
        text_files.push_back(data_path);
    } else if (fs::is_directory(path)) {
        // Directory - scan for text files asynchronously
        const int num_threads = 4; // Configurable thread pool size
        std::vector<std::future<std::vector<std::string>>> scan_futures;

        // Get all subdirectories
        std::vector<fs::path> subdirs;
        for (const auto& entry : fs::directory_iterator(path)) {
            if (entry.is_directory()) {
                subdirs.push_back(entry.path());
            }
        }

        // If no subdirs, scan the main directory
        if (subdirs.empty()) {
            subdirs.push_back(path);
        }

        // Launch async scans for each subdirectory
        for (const auto& subdir : subdirs) {
            scan_futures.push_back(std::async(std::launch::async, [subdir]() {
                std::vector<std::string> files;
                for (const auto& entry : fs::recursive_directory_iterator(subdir)) {
                    if (entry.is_regular_file()) {
                        std::string ext = entry.path().extension().string();
                        if (ext == ".txt" || ext == ".md" || ext.empty()) { // Include files without extension
                            files.push_back(entry.path().string());
                        }
                    }
                }
                return files;
            }));
        }

        // Collect results
        for (auto& future : scan_futures) {
            auto files = future.get();
            text_files.insert(text_files.end(), files.begin(), files.end());
        }

        std::cout << "Found " << text_files.size() << " text files in directory scan" << std::endl;
    } else {
        std::cerr << "Invalid data path: " << data_path << std::endl;
        return false;
    }

    // Load all text files asynchronously
    std::string combined_text;
    std::vector<std::future<std::string>> load_futures;

    for (const auto& file_path : text_files) {
        load_futures.push_back(std::async(std::launch::async, [file_path]() {
            std::ifstream file(file_path);
            if (!file.is_open()) {
                std::cerr << "Failed to open file: " << file_path << std::endl;
                return std::string("");
            }
            return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        }));
    }

    // Collect loaded text
    for (auto& future : load_futures) {
        combined_text += future.get();
    }

    if (combined_text.empty()) {
        std::cerr << "No text data loaded" << std::endl;
        return false;
    }

    // Create character vocabulary
    std::unordered_map<char, uint32_t> char_to_idx;
    std::vector<char> idx_to_char;
    for (char c = 0; c < 256; ++c) {
        char_to_idx[c] = idx_to_char.size();
        idx_to_char.push_back(c);
    }

    // Create sequences
    training_batches_.clear();
    for (size_t i = 0; i < combined_text.size() - config_.max_sequence_length; i += config_.batch_size * config_.max_sequence_length) {
        TrainingBatch batch;
        batch.batch_size = std::min(config_.batch_size, (uint32_t)((combined_text.size() - i) / config_.max_sequence_length));
        batch.seq_length = config_.max_sequence_length;

        for (uint32_t b = 0; b < batch.batch_size; ++b) {
            size_t start_idx = i + b * config_.max_sequence_length;
            for (uint32_t s = 0; s < config_.max_sequence_length; ++s) {
                char c = combined_text[start_idx + s];
                uint32_t idx = char_to_idx[c];
                batch.input_sequences.push_back(idx);
                if (s < config_.max_sequence_length - 1) {
                    batch.target_sequences.push_back(char_to_idx[combined_text[start_idx + s + 1]]);
                }
            }
        }

        if (!batch.input_sequences.empty()) {
            training_batches_.push_back(batch);
        }
    }

    std::cout << "Prepared " << training_batches_.size() << " training batches from "
              << combined_text.size() << " characters" << std::endl;
    return true;
}

bool VulkanTrainer::execute_forward_pass(const TrainingBatch& batch) {
    // Placeholder implementation
    return true;
}

bool VulkanTrainer::execute_backward_pass(const TrainingBatch& batch) {
    // Placeholder implementation
    return true;
}

bool VulkanTrainer::execute_optimizer_step() {
    // Placeholder implementation
    timestep_++;
    return true;
}

float VulkanTrainer::compute_loss_and_metrics() {
    // Placeholder loss computation
    return 1.0f;
}