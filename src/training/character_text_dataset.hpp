#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>
#include "../core/attention.hpp"  // For Tensor struct
#include "../utils/utf8_processor.hpp"  // For UTF-8 processing

namespace fs = std::filesystem;

/**
 * Character-Level Text Dataset for Language Training
 *
 * Handles loading and preprocessing of large text corpora at the character level.
 * Unlike token-based datasets, this works directly with UTF-8 characters,
 * providing true multilingual support without tokenization artifacts.
 */
class CharacterTextDataset {
public:
    CharacterTextDataset(const std::string& data_path, std::shared_ptr<UTF8Processor> utf8_processor);

    /**
     * Load character sequences from text files
     * @param max_length Maximum sequence length in characters
     * @return Vector of character sequences
     */
    std::vector<std::string> load_character_sequences(size_t max_length = 2048);

    /**
     * Create training pairs for next-character prediction
     * @param sequences Input character sequences
     * @param context_length Context window size
     * @return Vector of (input_sequence, target_sequence) pairs
     */
    std::vector<std::pair<std::string, std::string>> create_training_pairs(
        const std::vector<std::string>& sequences, size_t context_length);

    /**
     * Create batches for training
     * @param sequences Character sequences to batch
     * @param batch_size Batch size
     * @param max_length Maximum sequence length
     * @return Vector of batched tensors
     */
    std::vector<Tensor> create_character_batch(const std::vector<std::string>& sequences,
                                             int batch_size, int max_length);

    /**
     * Encode characters to tensor format for model input
     * @param text Input text
     * @return Tensor with character indices
     */
    Tensor encode_characters_to_tensor(const std::string& text);

    /**
     * Decode tensor back to characters
     * @param tensor Tensor with character indices
     * @return Decoded text
     */
    std::string decode_tensor_to_characters(const Tensor& tensor);

    /**
     * Get character vocabulary size
     * @return Number of unique characters
     */
    size_t get_vocab_size() const { return char_to_id_.size(); }

    /**
     * Get character-to-ID mapping
     * @return Const reference to character mapping
     */
    const std::unordered_map<uint32_t, int>& get_char_to_id() const { return char_to_id_; }

    /**
     * Get ID-to-character mapping
     * @return Const reference to ID mapping
     */
    const std::vector<uint32_t>& get_id_to_char() const { return id_to_char_; }

    /**
     * Save vocabulary to file
     * @param path File path to save vocabulary
     */
    void save_vocabulary(const std::string& path) const;

    /**
     * Load vocabulary from file
     * @param path File path to load vocabulary
     */
    void load_vocabulary(const std::string& path);

private:
    std::string data_path_;
    std::shared_ptr<UTF8Processor> utf8_processor_;

    // Character vocabulary mappings
    std::unordered_map<uint32_t, int> char_to_id_;
    std::vector<uint32_t> id_to_char_;

    // Special character constants
    static const char32_t UNKNOWN_CHAR;  // Unicode replacement character
    static const char32_t PADDING_CHAR = 0x0000;  // Null character for padding

    /**
     * Build character vocabulary from text data
     * @param text Input text to analyze
     */
    void build_vocabulary(const std::string& text);

    /**
     * Convert UTF-8 string to Unicode codepoints
     * @param utf8_str UTF-8 encoded string
     * @return Vector of Unicode codepoints
     */
    std::vector<uint32_t> utf8_to_codepoints(const std::string& utf8_str) const;

    /**
     * Convert Unicode codepoints to UTF-8 string
     * @param codepoints Vector of Unicode codepoints
     * @return UTF-8 encoded string
     */
    std::string codepoints_to_utf8(const std::vector<uint32_t>& codepoints) const;

    /**
     * Load text files from directory
     * @param directory Directory containing text files
     * @return Combined text content
     */
    std::string load_text_files(const std::string& directory);

    /**
     * Preprocess text (normalize, clean, etc.)
     * @param text Raw text input
     * @return Preprocessed text
     */
    std::string preprocess_text(const std::string& text) const;

    /**
     * Split text into sequences of specified length
     * @param text Input text
     * @param max_length Maximum sequence length
     * @return Vector of text sequences
     */
    std::vector<std::string> split_into_sequences(const std::string& text, size_t max_length);
};