#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

struct VulkanTrainingConfig {
    uint32_t max_sequence_length = 128;
    uint32_t vocab_size = 256;
    uint32_t batch_size = 16;
    uint32_t hidden_size = 512;
    uint32_t num_layers = 1;
    float learning_rate = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    uint32_t max_epochs = 100;
    uint32_t save_every_epochs = 10;
};

struct TrainingBatch {
    std::vector<float> input_sequences;
    std::vector<uint32_t> target_sequences;
    uint32_t batch_size;
    uint32_t seq_length;
};

class VulkanTrainer {
public:
    VulkanTrainer(VkDevice device, VkPhysicalDevice physical_device,
                 uint32_t compute_queue_family_index, VkQueue compute_queue,
                 const VulkanTrainingConfig& config);
    ~VulkanTrainer();

    bool initialize();
    bool load_training_data(const std::string& data_path);
    bool train_epoch();
    bool save_checkpoint(const std::string& checkpoint_path);
    bool load_checkpoint(const std::string& checkpoint_path);

    // Model management
    bool initialize_model();
    bool save_model(const std::string& model_path);
    bool load_model(const std::string& model_path);

    // Inference
    std::string generate_text(const std::string& prompt, uint32_t max_length = 100);

    // Training metrics
    float get_current_loss() const { return current_loss_; }
    float get_current_perplexity() const { return current_perplexity_; }
    uint32_t get_current_epoch() const { return current_epoch_; }

private:
    // Vulkan resources
    VkDevice device_;
    VkPhysicalDevice physical_device_;
    uint32_t compute_queue_family_index_;
    VkQueue compute_queue_;
    VkCommandPool command_pool_;
    VkDescriptorPool descriptor_pool_;
    VkDescriptorSetLayout descriptor_set_layout_;
    VkPipelineLayout pipeline_layout_;

    // Shaders and pipelines
    VkShaderModule linear_forward_shader_;
    VkShaderModule linear_backward_shader_;
    VkShaderModule adam_optimizer_shader_;
    VkShaderModule cross_entropy_loss_shader_;
    VkShaderModule gradient_accumulation_shader_;

    VkPipeline linear_forward_pipeline_;
    VkPipeline linear_backward_pipeline_;
    VkPipeline adam_optimizer_pipeline_;
    VkPipeline cross_entropy_loss_pipeline_;
    VkPipeline gradient_accumulation_pipeline_;

    // Model parameters and gradients
    std::vector<float> model_params_;
    std::vector<float> model_grads_;
    std::vector<float> adam_m_;
    std::vector<float> adam_v_;

    // Buffers
    VkBuffer param_buffer_;
    VkBuffer grad_buffer_;
    VkBuffer adam_m_buffer_;
    VkBuffer adam_v_buffer_;
    VkBuffer input_buffer_;
    VkBuffer target_buffer_;
    VkBuffer loss_buffer_;

    VkDeviceMemory param_memory_;
    VkDeviceMemory grad_memory_;
    VkDeviceMemory adam_m_memory_;
    VkDeviceMemory adam_v_memory_;
    VkDeviceMemory input_memory_;
    VkDeviceMemory target_memory_;
    VkDeviceMemory loss_memory_;

    // Training data
    std::vector<TrainingBatch> training_batches_;
    VulkanTrainingConfig config_;

    // Model architecture
    std::vector<float> embedding_weights_;  // vocab_size x hidden_size
    std::vector<float> hidden_weights_;     // hidden_size x hidden_size
    std::vector<float> hidden_bias_;        // hidden_size
    std::vector<float> output_weights_;     // hidden_size x vocab_size
    std::vector<float> output_bias_;        // vocab_size

    // Training state
    uint32_t current_epoch_;
    float current_loss_;
    float current_perplexity_;
    uint32_t timestep_;  // For Adam

    // Private methods
    bool create_shaders();
    bool create_pipelines();
    bool create_buffers();
    bool create_descriptor_sets();
    bool prepare_training_data(const std::string& data_path);
    bool execute_forward_pass(const TrainingBatch& batch);
    bool execute_backward_pass(const TrainingBatch& batch);
    bool execute_optimizer_step();
    float compute_loss_and_metrics();

    // Vulkan utility functions
    VkShaderModule create_shader_module(const std::vector<uint32_t>& code);
    VkPipeline create_compute_pipeline(VkShaderModule shader, const std::string& entry_point);
    bool create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                      VkBuffer& buffer, VkDeviceMemory& memory);
    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);
    std::vector<uint32_t> load_spirv_file(const std::string& filename);
    bool create_descriptor_set_layout();
    bool create_pipeline_layout();
    VkDescriptorSet allocate_descriptor_set();
    void update_descriptor_set(VkDescriptorSet descriptor_set, VkBuffer buffer, uint32_t binding);
    VkCommandBuffer begin_command_buffer();
    void submit_command_buffer(VkCommandBuffer command_buffer);
    void copy_data_to_buffer(VkBuffer buffer, const void* data, VkDeviceSize size);
    void copy_data_from_buffer(VkBuffer buffer, void* data, VkDeviceSize size);
};