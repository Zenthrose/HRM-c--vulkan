#include "character_text_dataset.hpp"
#include <codecvt>
#include <locale>
#include <sstream>
#include <regex>
#include <filesystem>

// Define the static constant
const char32_t CharacterTextDataset::UNKNOWN_CHAR = 0xFFFD;

CharacterTextDataset::CharacterTextDataset(const std::string& data_path, std::shared_ptr<UTF8Processor> utf8_processor)
    : data_path_(data_path), utf8_processor_(utf8_processor) {

    // Initialize with basic ASCII characters
    for (        uint32_t c = 32; c < 127; ++c) {  // Printable ASCII
        char_to_id_[c] = id_to_char_.size();
        id_to_char_.push_back(c);
    }

    // Add some common Unicode characters
    std::vector<uint32_t> common_unicode = {
        0x00A0,  // Non-breaking space
        0x00AD,  // Soft hyphen
        0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200A, // Spaces
        0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, // Hyphens and dashes
        0x2018, 0x2019, 0x201A, 0x201B, 0x201C, 0x201D, 0x201E, 0x201F, // Quotes
        0x2026, // Ellipsis
        0x2030, // Per mille sign
    };

    for (uint32_t c : common_unicode) {
        if (char_to_id_.find(c) == char_to_id_.end()) {
            char_to_id_[c] = id_to_char_.size();
            id_to_char_.push_back(c);
        }
    }

    std::cout << "CharacterTextDataset initialized with " << get_vocab_size() << " base characters" << std::endl;
}

std::vector<std::string> CharacterTextDataset::load_character_sequences(size_t max_length) {
    std::vector<std::string> sequences;

    try {
        std::string text = load_text_files(data_path_);
        text = preprocess_text(text);

        // Build vocabulary from the actual text
        build_vocabulary(text);

        // Split into sequences
        sequences = split_into_sequences(text, max_length);

        std::cout << "Loaded " << sequences.size() << " character sequences from " << data_path_ << std::endl;
        std::cout << "Vocabulary size: " << get_vocab_size() << " characters" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error loading character sequences: " << e.what() << std::endl;
    }

    return sequences;
}

std::vector<std::pair<std::string, std::string>> CharacterTextDataset::create_training_pairs(
    const std::vector<std::string>& sequences, size_t context_length) {

    std::vector<std::pair<std::string, std::string>> training_pairs;

    for (const std::string& sequence : sequences) {
        if (sequence.length() <= context_length) continue;

        // Create sliding window of context_length characters
        for (size_t i = 0; i <= sequence.length() - context_length - 1; ++i) {
            std::string input = sequence.substr(i, context_length);
            std::string target = sequence.substr(i + 1, context_length);  // Next character prediction

            training_pairs.emplace_back(input, target);
        }
    }

    std::cout << "Created " << training_pairs.size() << " training pairs with context length " << context_length << std::endl;
    return training_pairs;
}

std::vector<Tensor> CharacterTextDataset::create_character_batch(
    const std::vector<std::string>& sequences, int batch_size, int max_length) {

    std::vector<Tensor> batches;

    for (size_t batch_start = 0; batch_start < sequences.size(); batch_start += batch_size) {
        size_t current_batch_size = std::min(static_cast<size_t>(batch_size),
                                           sequences.size() - batch_start);

        // Create batch tensor: (batch_size, max_length)
        Tensor batch;
        batch.shape = {static_cast<uint32_t>(current_batch_size), static_cast<uint32_t>(max_length)};
        batch.data.resize(current_batch_size * max_length, 0.0f);  // Initialize with padding

        for (size_t i = 0; i < current_batch_size; ++i) {
            const std::string& sequence = sequences[batch_start + i];
            std::vector<uint32_t> codepoints = utf8_to_codepoints(sequence);

            // Convert to character IDs and pad/truncate
            for (size_t j = 0; j < max_length && j < codepoints.size(); ++j) {
                uint32_t c = codepoints[j];
                auto it = char_to_id_.find(c);
                int char_id = (it != char_to_id_.end()) ? it->second : char_to_id_[UNKNOWN_CHAR];

                batch.data[i * max_length + j] = static_cast<float>(char_id);
            }
        }

        batches.push_back(batch);
    }

    return batches;
}

Tensor CharacterTextDataset::encode_characters_to_tensor(const std::string& text) {
    std::vector<uint32_t> codepoints = utf8_to_codepoints(text);

    Tensor tensor;
    tensor.shape = {1, static_cast<uint32_t>(codepoints.size())};  // (1, seq_len)
    tensor.data.resize(codepoints.size());

    for (size_t i = 0; i < codepoints.size(); ++i) {
        uint32_t c = codepoints[i];
        auto it = char_to_id_.find(c);
        int char_id = (it != char_to_id_.end()) ? it->second : char_to_id_[UNKNOWN_CHAR];
        tensor.data[i] = static_cast<float>(char_id);
    }

    return tensor;
}

std::string CharacterTextDataset::decode_tensor_to_characters(const Tensor& tensor) {
    if (tensor.shape.size() != 2 || tensor.shape[0] != 1) {
        throw std::invalid_argument("Tensor must be (1, seq_len) for decoding");
    }

    std::vector<uint32_t> codepoints;
    for (float val : tensor.data) {
        int char_id = static_cast<int>(val);
        if (char_id >= 0 && char_id < static_cast<int>(id_to_char_.size())) {
            codepoints.push_back(id_to_char_[char_id]);
        } else {
            // Handle out-of-bounds character IDs gracefully
            codepoints.push_back('?');  // Use question mark for unknown characters
        }
    }

    return codepoints_to_utf8(codepoints);
}

void CharacterTextDataset::save_vocabulary(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open vocabulary file for writing: " + path);
    }

    // Write vocabulary size
    size_t vocab_size = id_to_char_.size();
    file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));

    // Write character mappings
    for (char32_t c : id_to_char_) {
        file.write(reinterpret_cast<const char*>(&c), sizeof(c));
    }

    std::cout << "Saved vocabulary with " << vocab_size << " characters to " << path << std::endl;
}

void CharacterTextDataset::load_vocabulary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open vocabulary file for reading: " + path);
    }

    // Read vocabulary size
    size_t vocab_size;
    file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));

    // Read character mappings
    id_to_char_.resize(vocab_size);
    char_to_id_.clear();

    for (size_t i = 0; i < vocab_size; ++i) {
        char32_t c;
        file.read(reinterpret_cast<char*>(&c), sizeof(c));
        id_to_char_[i] = c;
        char_to_id_[c] = i;
    }

    std::cout << "Loaded vocabulary with " << vocab_size << " characters from " << path << std::endl;
}

// Private helper methods

void CharacterTextDataset::build_vocabulary(const std::string& text) {
    std::vector<uint32_t> codepoints = utf8_to_codepoints(text);

    for (uint32_t c : codepoints) {
        if (char_to_id_.find(c) == char_to_id_.end()) {
            char_to_id_[c] = id_to_char_.size();
            id_to_char_.push_back(c);
        }
    }
}

std::vector<uint32_t> CharacterTextDataset::utf8_to_codepoints(const std::string& utf8_str) const {
    // Use UTF8Processor for consistent UTF-8 handling
    if (utf8_processor_) {
        return utf8_processor_->encode_utf8(utf8_str);
    }

    // Fallback: simple byte processing (should not be used in normal operation)
    std::vector<uint32_t> codepoints;
    for (unsigned char c : utf8_str) {
        codepoints.push_back(static_cast<uint32_t>(c));
    }
    return codepoints;
}

std::string CharacterTextDataset::codepoints_to_utf8(const std::vector<uint32_t>& codepoints) const {
    // Use UTF8Processor for consistent UTF-8 handling
    if (utf8_processor_) {
        return utf8_processor_->decode_utf8(codepoints);
    }

    // Fallback: simple byte conversion (should not be used in normal operation)
    std::string utf8_str;
    for (uint32_t codepoint : codepoints) {
        if (codepoint <= 0xFF) {
            utf8_str.push_back(static_cast<char>(codepoint));
        } else {
            utf8_str.push_back('?'); // Replacement for unsupported codepoints
        }
    }
    return utf8_str;
}

std::string CharacterTextDataset::load_text_files(const std::string& directory) {
    std::string combined_text;

    try {
        if (fs::is_directory(directory)) {
            // Load all .txt files from directory
            for (const auto& entry : fs::directory_iterator(directory)) {
                if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                    std::ifstream file(entry.path());
                    if (file.is_open()) {
                        std::string content((std::istreambuf_iterator<char>(file)),
                                          std::istreambuf_iterator<char>());
                        combined_text += content + "\n";
                        file.close();
                    }
                }
            }
        } else if (fs::is_regular_file(directory)) {
            // Load single file
            std::ifstream file(directory);
            if (file.is_open()) {
                combined_text = std::string((std::istreambuf_iterator<char>(file)),
                                          std::istreambuf_iterator<char>());
                file.close();
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading text files: " << e.what() << std::endl;
    }

    return combined_text;
}

std::string CharacterTextDataset::preprocess_text(const std::string& text) const {
    std::string processed = text;

    // Normalize line endings
    std::regex crlf("\r\n");
    processed = std::regex_replace(processed, crlf, "\n");

    // Remove excessive whitespace (more than 2 consecutive newlines)
    std::regex excessive_newlines("\n{3,}");
    processed = std::regex_replace(processed, excessive_newlines, "\n\n");

    // Trim leading/trailing whitespace from each line
    std::istringstream iss(processed);
    std::string line;
    std::string result;

    while (std::getline(iss, line)) {
        // Trim whitespace from line
        line.erase(line.begin(), std::find_if(line.begin(), line.end(),
                  [](int ch) { return !std::isspace(ch); }));
        line.erase(std::find_if(line.rbegin(), line.rend(),
                  [](int ch) { return !std::isspace(ch); }).base(), line.end());

        if (!line.empty()) {
            result += line + "\n";
        }
    }

    return result;
}

std::vector<std::string> CharacterTextDataset::split_into_sequences(const std::string& text, size_t max_length) {
    std::vector<std::string> sequences;

    // Split text into chunks of max_length characters
    for (size_t i = 0; i < text.length(); i += max_length) {
        size_t chunk_size = std::min(max_length, text.length() - i);
        sequences.push_back(text.substr(i, chunk_size));
    }

    return sequences;
}