#include "utf8_processor.hpp"
#include <iostream>
#include <random>
#include <algorithm>

UTF8Processor::UTF8Processor() : config_{} {
    config_.max_sequence_length = 1024;
    config_.embedding_dim = 768;
    config_.use_byte_fallback = true;
#ifndef NO_VULKAN
    config_.physicalDevice = VK_NULL_HANDLE;
    config_.device = VK_NULL_HANDLE;
    config_.computeQueue = VK_NULL_HANDLE;
    config_.computeQueueFamilyIndex = 0;
    config_.commandPool = VK_NULL_HANDLE;
#endif
    initialize_char_embeddings();
}

UTF8Processor::UTF8Processor(const UTF8Config& config) : config_(config) {
    std::cout << "Initializing UTF-8 Processor..." << std::endl;
    initialize_char_embeddings();
    std::cout << "UTF-8 Processor initialized" << std::endl;
}

std::vector<uint32_t> UTF8Processor::encode_utf8(const std::string& text) {
    std::vector<uint32_t> codes;
    size_t pos = 0;

    while (pos < text.length()) {
        uint32_t codepoint = decode_utf8_codepoint(text, pos);
        codes.push_back(codepoint);
    }

    // Truncate if too long
    if (codes.size() > config_.max_sequence_length) {
        codes.resize(config_.max_sequence_length);
    }

    return codes;
}

std::string UTF8Processor::decode_utf8(const std::vector<uint32_t>& codes) {
    std::string result;

    for (uint32_t codepoint : codes) {
        result += encode_utf8_codepoint(codepoint);
    }

    return result;
}

Tensor UTF8Processor::process_characters(const std::string& text) {
    auto codes = encode_utf8(text);
    Tensor result;

    // Create embeddings for each character
    for (size_t i = 0; i < codes.size(); ++i) {
        auto embedding = get_char_embedding(codes[i]);
        result.data.insert(result.data.end(), embedding.begin(), embedding.end());
    }

    // Pad to max length if needed
    while (result.data.size() < config_.max_sequence_length * config_.embedding_dim) {
        result.data.push_back(0.0f);
    }

    return result;
}

std::string UTF8Processor::generate_text(const Tensor& hidden_states) {
    // Simplified: convert hidden states back to characters
    // In a real implementation, this would use the language model head
    std::vector<uint32_t> codes;

    for (size_t i = 0; i < hidden_states.data.size(); i += config_.embedding_dim) {
        // Find closest character embedding (simplified)
        uint32_t best_codepoint = 32; // default to space
        float best_distance = std::numeric_limits<float>::max();

        for (const auto& pair : char_embeddings_) {
            float distance = 0.0f;
            for (size_t j = 0; j < config_.embedding_dim; ++j) {
                float diff = hidden_states.data[i + j] - pair.second[j];
                distance += diff * diff;
            }

            if (distance < best_distance) {
                best_distance = distance;
                best_codepoint = pair.first;
            }
        }

        codes.push_back(best_codepoint);
    }

    return decode_utf8(codes);
}

bool UTF8Processor::is_valid_utf8(const std::string& text) {
    size_t pos = 0;
    while (pos < text.length()) {
        try {
            decode_utf8_codepoint(text, pos);
        } catch (...) {
            return false;
        }
    }
    return true;
}

std::vector<size_t> UTF8Processor::find_invalid_sequences(const std::string& text) {
    std::vector<size_t> invalid_positions;
    size_t pos = 0;

    while (pos < text.length()) {
        size_t start_pos = pos;
        try {
            decode_utf8_codepoint(text, pos);
        } catch (...) {
            invalid_positions.push_back(start_pos);
            pos = start_pos + 1; // Skip invalid byte
        }
    }

    return invalid_positions;
}

int UTF8Processor::get_utf8_char_length(unsigned char first_byte) {
    // Static implementation for UTF-8 character length detection
    if ((first_byte & 0x80) == 0) return 1;        // 0xxxxxxx
    if ((first_byte & 0xE0) == 0xC0) return 2;     // 110xxxxx
    if ((first_byte & 0xF0) == 0xE0) return 3;     // 1110xxxx
    if ((first_byte & 0xF8) == 0xF0) return 4;     // 11110xxx
    return -1; // Invalid
}

bool UTF8Processor::is_utf8_continuation_byte(unsigned char byte) {
    return (byte & 0xC0) == 0x80; // 10xxxxxx
}

uint32_t UTF8Processor::decode_utf8_codepoint(const std::string& bytes, size_t& pos) {
    if (pos >= bytes.length()) {
        throw std::runtime_error("Unexpected end of UTF-8 sequence");
    }

    unsigned char first = bytes[pos];
    int length = get_utf8_char_length(first);

    if (length == -1 || pos + length > bytes.length()) {
        throw std::runtime_error("Invalid UTF-8 sequence");
    }

    uint32_t codepoint = 0;

    if (length == 1) {
        codepoint = first;
    } else {
        // Extract bits from first byte
        codepoint = first & ((1 << (7 - length)) - 1);

        // Extract bits from continuation bytes
        for (int i = 1; i < length; ++i) {
            unsigned char cont = bytes[pos + i];
            if (!is_utf8_continuation_byte(cont)) {
                throw std::runtime_error("Invalid UTF-8 continuation byte");
            }
            codepoint = (codepoint << 6) | (cont & 0x3F);
        }
    }

    pos += length;
    return codepoint;
}

std::string UTF8Processor::encode_utf8_codepoint(uint32_t codepoint) {
    std::string result;

    if (codepoint <= 0x7F) {
        // 1 byte: 0xxxxxxx
        result.push_back(codepoint);
    } else if (codepoint <= 0x7FF) {
        // 2 bytes: 110xxxxx 10xxxxxx
        result.push_back(0xC0 | (codepoint >> 6));
        result.push_back(0x80 | (codepoint & 0x3F));
    } else if (codepoint <= 0xFFFF) {
        // 3 bytes: 1110xxxx 10xxxxxx 10xxxxxx
        result.push_back(0xE0 | (codepoint >> 12));
        result.push_back(0x80 | ((codepoint >> 6) & 0x3F));
        result.push_back(0x80 | (codepoint & 0x3F));
    } else if (codepoint <= 0x10FFFF) {
        // 4 bytes: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
        result.push_back(0xF0 | (codepoint >> 18));
        result.push_back(0x80 | ((codepoint >> 12) & 0x3F));
        result.push_back(0x80 | ((codepoint >> 6) & 0x3F));
        result.push_back(0x80 | (codepoint & 0x3F));
    } else {
        throw std::runtime_error("Invalid Unicode codepoint");
    }

    return result;
}

void UTF8Processor::initialize_char_embeddings() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);

    // Initialize embeddings for common characters
    for (uint32_t codepoint = 0; codepoint < 256; ++codepoint) {
        std::vector<float> embedding(config_.embedding_dim);
        for (int i = 0; i < config_.embedding_dim; ++i) {
            embedding[i] = dist(gen);
        }
        char_embeddings_[codepoint] = embedding;
    }

    // Special embeddings for extended Unicode
    for (uint32_t codepoint = 256; codepoint < 1024; ++codepoint) {
        std::vector<float> embedding(config_.embedding_dim);
        for (int i = 0; i < config_.embedding_dim; ++i) {
            embedding[i] = dist(gen) * 0.5f; // Slightly different distribution
        }
        char_embeddings_[codepoint] = embedding;
    }
}

std::vector<float> UTF8Processor::get_char_embedding(uint32_t codepoint) {
    auto it = char_embeddings_.find(codepoint);
    if (it != char_embeddings_.end()) {
        return it->second;
    }

    // Fallback: create embedding on the fly
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);

    std::vector<float> embedding(config_.embedding_dim);
    for (int i = 0; i < config_.embedding_dim; ++i) {
        embedding[i] = dist(gen);
    }

    char_embeddings_[codepoint] = embedding;
    return embedding;
}

// CPU-based character processing for offloading
bool UTF8Processor::validate_utf8_cpu(const std::string& text) {
    size_t i = 0;
    while (i < text.size()) {
        int char_len = get_utf8_char_length(static_cast<unsigned char>(text[i]));
        if (char_len == 0 || i + char_len > text.size()) {
            return false;
        }
        for (int j = 1; j < char_len; ++j) {
            if (!is_utf8_continuation_byte(static_cast<unsigned char>(text[i + j]))) {
                return false;
            }
        }
        i += char_len;
    }
    return true;
}

std::string UTF8Processor::normalize_characters_cpu(const std::string& text) {
    std::string normalized;
    size_t i = 0;
    while (i < text.size()) {
        size_t pos = i;
        uint32_t codepoint = decode_utf8_codepoint(text, pos);
        if (codepoint == 0) break;

        // Basic normalization: convert common variants
        if (codepoint >= 0x41 && codepoint <= 0x5A) { // A-Z -> a-z
            codepoint += 0x20;
        }
        // Could add more normalization rules here

        normalized += encode_utf8_codepoint(codepoint);
        i = pos;
    }
    return normalized;
}

std::vector<uint32_t> UTF8Processor::encode_characters_cpu(const std::string& text) {
    std::vector<uint32_t> codes;
    size_t i = 0;
    while (i < text.size()) {
        size_t pos = i;
        uint32_t codepoint = decode_utf8_codepoint(text, pos);
        if (codepoint == 0) break;
        codes.push_back(codepoint);
        i = pos;
    }
    return codes;
}

std::string UTF8Processor::decode_characters_cpu(const std::vector<uint32_t>& codes) {
    std::string text;
    for (uint32_t codepoint : codes) {
        text += encode_utf8_codepoint(codepoint);
    }
    return text;
}