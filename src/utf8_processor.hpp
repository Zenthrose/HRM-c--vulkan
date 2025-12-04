#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include "attention.hpp" // For Tensor

struct UTF8Config {
    int max_sequence_length;
    int embedding_dim;
    bool use_byte_fallback;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue computeQueue;
    uint32_t computeQueueFamilyIndex;
    VkCommandPool commandPool;
};

class UTF8Processor {
public:
    UTF8Processor(const UTF8Config& config);
    ~UTF8Processor() = default;

    // UTF-8 encoding/decoding
    std::vector<uint32_t> encode_utf8(const std::string& text);
    std::string decode_utf8(const std::vector<uint32_t>& codes);

    // Character-level processing
    Tensor process_characters(const std::string& text);
    std::string generate_text(const Tensor& hidden_states);

    // Validation
    bool is_valid_utf8(const std::string& text);
    std::vector<size_t> find_invalid_sequences(const std::string& text);

private:
    UTF8Config config_;

    // UTF-8 utilities
    int get_utf8_char_length(unsigned char first_byte);
    bool is_utf8_continuation_byte(unsigned char byte);
    uint32_t decode_utf8_codepoint(const std::string& bytes, size_t& pos);
    std::string encode_utf8_codepoint(uint32_t codepoint);

    // Character embeddings (simplified - could be learned)
    std::unordered_map<uint32_t, std::vector<float>> char_embeddings_;

    // Initialize basic character embeddings
    void initialize_char_embeddings();
    std::vector<float> get_char_embedding(uint32_t codepoint);
};