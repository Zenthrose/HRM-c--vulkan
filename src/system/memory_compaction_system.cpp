#include "memory_compaction_system.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <iomanip>
#include <unordered_set>
#include <set>
#include <regex>

MemoryCompactionSystem::MemoryCompactionSystem(const MemoryCompactionConfig& config)
    : config_(config), auto_compaction_running_(false) {

    cloud_storage_manager_ = config.cloud_storage_manager;

    // Create compaction directory if it doesn't exist
    fs::create_directories(config.compaction_directory);

    std::cout << "Memory Compaction System initialized" << std::endl;
    std::cout << "Compaction directory: " << config.compaction_directory << std::endl;
    if (cloud_storage_manager_) {
        std::cout << "Cloud storage integration enabled" << std::endl;
    }
}

MemoryCompactionSystem::~MemoryCompactionSystem() {
    if (auto_compaction_running_) {
        auto_compaction_running_ = false;
        if (auto_compaction_thread_.joinable()) {
            auto_compaction_thread_.join();
        }
    }
}

MemoryCompactionResult MemoryCompactionSystem::compact_memory(
    const std::vector<ConversationEntry>& entries,
    MemoryCompactionLevel level) {

    auto start_time = std::chrono::high_resolution_clock::now();

    MemoryCompactionResult result;
    result.success = false;
    result.original_size = calculate_memory_usage(entries);

    if (entries.empty()) {
        result.error_message = "No entries to compact";
        return result;
    }

    try {
        // Apply compaction based on level
        std::vector<uint8_t> compacted_data;
        switch (level) {
            case MemoryCompactionLevel::LIGHT:
                compacted_data = apply_light_compaction(entries);
                break;
            case MemoryCompactionLevel::MEDIUM:
                compacted_data = apply_medium_compaction(entries);
                break;
            case MemoryCompactionLevel::HEAVY:
                compacted_data = apply_heavy_compaction(entries);
                break;
            case MemoryCompactionLevel::EXTREME:
                compacted_data = apply_extreme_compaction(entries);
                break;
        }

        // Compress the data
        std::vector<uint8_t> compressed_data;
        switch (config_.preferred_algorithm) {
            case CompressionAlgorithm::LZ4:
                compressed_data = compress_lz4(compacted_data);
                break;
            case CompressionAlgorithm::ZSTD:
                compressed_data = compress_zstd(compacted_data);
                break;
            case CompressionAlgorithm::GZIP:
                compressed_data = compress_gzip(compacted_data);
                break;
            case CompressionAlgorithm::BROTLI:
                compressed_data = compress_brotli(compacted_data);
                break;
            case CompressionAlgorithm::NONE:
                compressed_data = compacted_data;
                break;
        }

        // Create compacted memory object
        CompactedMemory compaction;
        compaction.compaction_id = generate_compaction_id();
        compaction.created_time = std::chrono::system_clock::now();
        compaction.level = level;
        compaction.algorithm = config_.preferred_algorithm;
        compaction.compressed_data = compressed_data;
        compaction.original_size_bytes = result.original_size;
        compaction.compressed_size_bytes = compressed_data.size();
        compaction.compression_ratio = static_cast<double>(compaction.original_size_bytes) /
                                     std::max(size_t(1), compaction.compressed_size_bytes);

        // Extract entry IDs
        for (const auto& entry : entries) {
            compaction.entry_ids.push_back(entry.id);
        }

        // Add metadata
        compaction.metadata["compaction_level"] = std::to_string(static_cast<int>(level));
        compaction.metadata["algorithm"] = std::to_string(static_cast<int>(config_.preferred_algorithm));
        compaction.metadata["entry_count"] = std::to_string(entries.size());

        // Save to file
        if (save_compaction_to_file(compaction)) {
            compacted_memories_[compaction.compaction_id] = compaction;

            // Upload to cloud storage if available
            if (cloud_storage_manager_) {
                upload_compaction_to_cloud(compaction);
            }

            result.success = true;
            result.compaction_id = compaction.compaction_id;
            result.compressed_size = compaction.compressed_size_bytes;
            result.compression_ratio = compaction.compression_ratio;
        } else {
            result.error_message = "Failed to save compaction to file";
        }

    } catch (const std::exception& e) {
        result.error_message = std::string("Compaction failed: ") + e.what();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    return result;
}

MemoryCompactionResult MemoryCompactionSystem::compact_memory_auto() {
    // Select entries for compaction
    auto entries_to_compact = select_entries_for_compaction();

    if (entries_to_compact.empty()) {
        MemoryCompactionResult result;
        result.success = false;
        result.error_message = "No entries available for compaction";
        return result;
    }

    return compact_memory(entries_to_compact, config_.default_level);
}

std::vector<ConversationEntry> MemoryCompactionSystem::decompress_memory(const std::string& compaction_id) {
    std::vector<ConversationEntry> entries;

    auto it = compacted_memories_.find(compaction_id);
    if (it == compacted_memories_.end()) {
        // Try to load from file
        CompactedMemory compaction = load_compaction_from_file(compaction_id);
        if (compaction.compaction_id.empty()) {
            // Try to download from cloud storage
            if (cloud_storage_manager_) {
                compaction = download_compaction_from_cloud(compaction_id);
                if (compaction.compaction_id.empty()) {
                    return entries; // Empty result
                }
                // Save locally for future use
                save_compaction_to_file(compaction);
            } else {
                return entries; // Empty result
            }
        }
        it = compacted_memories_.insert({compaction_id, compaction}).first;
    }

    const CompactedMemory& compaction = it->second;

    // Perform cleanup after memory access
    perform_memory_cleanup();

    try {
        // Decompress the data
        std::vector<uint8_t> decompressed_data;
        switch (compaction.algorithm) {
            case CompressionAlgorithm::LZ4:
                decompressed_data = decompress_lz4(compaction.compressed_data);
                break;
            case CompressionAlgorithm::ZSTD:
                decompressed_data = decompress_zstd(compaction.compressed_data);
                break;
            case CompressionAlgorithm::GZIP:
                decompressed_data = decompress_gzip(compaction.compressed_data);
                break;
            case CompressionAlgorithm::BROTLI:
                decompressed_data = decompress_brotli(compaction.compressed_data);
                break;
            case CompressionAlgorithm::NONE:
                decompressed_data = compaction.compressed_data;
                break;
        }

        // Deserialize entries
        entries = deserialize_entries(decompressed_data);

    } catch (const std::exception& e) {
        std::cerr << "Failed to decompress memory: " << e.what() << std::endl;
    }

    return entries;
}

std::vector<ConversationEntry> MemoryCompactionSystem::retrieve_recent_conversations(std::chrono::hours window) {
    auto cutoff_time = std::chrono::system_clock::now() - window;

    // Return recent conversations that haven't been compacted
    std::vector<ConversationEntry> recent;
    for (const auto& entry : recent_conversations_) {
        if (entry.timestamp > cutoff_time) {
            recent.push_back(entry);
        }
    }

    return recent;
}

std::vector<std::string> MemoryCompactionSystem::list_compactions() const {
    std::vector<std::string> ids;
    for (const auto& pair : compacted_memories_) {
        ids.push_back(pair.first);
    }
    return ids;
}

CompactedMemory MemoryCompactionSystem::get_compaction_info(const std::string& compaction_id) const {
    auto it = compacted_memories_.find(compaction_id);
    if (it != compacted_memories_.end()) {
        return it->second;
    }

    // Try to load from file
    return load_compaction_from_file(compaction_id);
}

bool MemoryCompactionSystem::delete_compaction(const std::string& compaction_id) {
    // Remove from memory
    compacted_memories_.erase(compaction_id);

    // Remove from disk
    std::string file_path = get_compaction_file_path(compaction_id);
    if (fs::exists(file_path)) {
        fs::remove(file_path);
    }

    // Remove from cloud storage
    if (cloud_storage_manager_) {
        cloud_storage_manager_->delete_compacted_memory(compaction_id, config_.default_cloud_provider);
    }

    return true;
}

void MemoryCompactionSystem::cleanup_old_compactions(std::chrono::hours max_age_hours) {
    auto cutoff_time = std::chrono::system_clock::now() - max_age_hours;

    std::vector<std::string> to_delete;
    for (const auto& pair : compacted_memories_) {
        if (pair.second.created_time < cutoff_time) {
            to_delete.push_back(pair.first);
        }
    }

    for (const auto& id : to_delete) {
        delete_compaction(id);
    }

    std::cout << "Cleaned up " << to_delete.size() << " old compactions" << std::endl;
}

size_t MemoryCompactionSystem::get_current_memory_usage() const {
    size_t total_size = 0;

    // Calculate size of recent conversations
    for (const auto& entry : recent_conversations_) {
        total_size += entry.original_size_bytes;
    }

    // Add size of compacted memories (compressed size)
    for (const auto& pair : compacted_memories_) {
        total_size += pair.second.compressed_size_bytes;
    }

    return total_size;
}

size_t MemoryCompactionSystem::get_compacted_memory_size() const {
    size_t total = 0;
    for (const auto& pair : compacted_memories_) {
        total += pair.second.compressed_size_bytes;
    }
    return total;
}

double MemoryCompactionSystem::get_average_compression_ratio() const {
    if (compacted_memories_.empty()) return 1.0;

    double total_ratio = 0.0;
    for (const auto& pair : compacted_memories_) {
        total_ratio += pair.second.compression_ratio;
    }

    return total_ratio / compacted_memories_.size();
}

std::unordered_map<std::string, size_t> MemoryCompactionSystem::get_memory_stats() const {
    std::unordered_map<std::string, size_t> stats;

    stats["recent_conversations"] = recent_conversations_.size();
    stats["compacted_memories"] = compacted_memories_.size();
    stats["current_memory_usage_mb"] = get_current_memory_usage() / (1024 * 1024);
    stats["compacted_memory_mb"] = get_compacted_memory_size() / (1024 * 1024);

    return stats;
}

void MemoryCompactionSystem::update_config(const MemoryCompactionConfig& new_config) {
    config_ = new_config;
}

MemoryCompactionConfig MemoryCompactionSystem::get_config() const {
    return config_;
}

void MemoryCompactionSystem::enable_auto_compaction(bool enable) {
    config_.auto_compaction_enabled = enable;

    if (enable && !auto_compaction_running_) {
        auto_compaction_running_ = true;
        auto_compaction_thread_ = std::thread(&MemoryCompactionSystem::auto_compaction_loop, this);
    } else if (!enable && auto_compaction_running_) {
        auto_compaction_running_ = false;
        if (auto_compaction_thread_.joinable()) {
            auto_compaction_thread_.join();
        }
    }
}

bool MemoryCompactionSystem::is_auto_compaction_enabled() const {
    return config_.auto_compaction_enabled;
}

void MemoryCompactionSystem::trigger_auto_compaction() {
    if (should_trigger_compaction()) {
        perform_auto_compaction();
    }
}

void MemoryCompactionSystem::perform_memory_cleanup() {
    // Clean up old compactions based on age
    cleanup_old_compactions(std::chrono::hours(24 * 30)); // 30 days

    // Clean up based on count if too many local compactions
    if (compacted_memories_.size() > 100) { // Keep max 100 compactions locally
        // Sort by age and keep only the newest 50
        std::vector<std::pair<std::string, std::chrono::system_clock::time_point>> compactions;
        for (const auto& pair : compacted_memories_) {
            compactions.emplace_back(pair.first, pair.second.created_time);
        }

        std::sort(compactions.begin(), compactions.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; }); // Newest first

        // Delete older ones beyond the limit
        for (size_t i = 50; i < compactions.size(); ++i) {
            delete_compaction(compactions[i].first);
        }
    }

    std::cout << "Memory cleanup completed" << std::endl;
}

void MemoryCompactionSystem::set_cleanup_thresholds(size_t max_local_compactions, std::chrono::hours max_age_hours) {
    // This could be used to configure cleanup parameters
    // For now, we'll use hardcoded values in perform_memory_cleanup
}

// Private methods - Compression algorithms (simplified implementations)

std::vector<uint8_t> MemoryCompactionSystem::compress_lz4(const std::vector<uint8_t>& data) {
    // Simple Run-Length Encoding (RLE) compression as a fallback
    // When LZ4 is not available, use this basic compression
    std::vector<uint8_t> compressed;
    
    if (data.empty()) return compressed;
    
    // RLE encoding: byte, run_length format
    for (size_t i = 0; i < data.size(); ++i) {
        uint8_t current = data[i];
        size_t run_length = 1;
        
        // Count consecutive identical bytes
        while (i + run_length < data.size() && data[i + run_length] == current && run_length < 255) {
            run_length++;
        }
        
        // Encode: run_length, byte
        if (run_length > 1 || current >= 128) {  // Encode repeats or high bytes
            compressed.push_back(255);  // Escape marker
            compressed.push_back(current);
            compressed.push_back(static_cast<uint8_t>(run_length));
        } else {
            compressed.push_back(current);
        }
        
        i += run_length - 1;
    }
    
    return compressed;
}

std::vector<uint8_t> MemoryCompactionSystem::compress_zstd(const std::vector<uint8_t>& data) {
    // Use LZ4 implementation as fallback for ZSTD
    return compress_lz4(data);
}

std::vector<uint8_t> MemoryCompactionSystem::compress_gzip(const std::vector<uint8_t>& data) {
    // Use LZ4 implementation as fallback for gzip
    return compress_lz4(data);
}

std::vector<uint8_t> MemoryCompactionSystem::compress_brotli(const std::vector<uint8_t>& data) {
    // Use LZ4 implementation as fallback for Brotli
    return compress_lz4(data);
}

std::vector<uint8_t> MemoryCompactionSystem::decompress_lz4(const std::vector<uint8_t>& compressed_data) {
    // RLE decompression - reverse of compress_lz4
    std::vector<uint8_t> decompressed;
    
    for (size_t i = 0; i < compressed_data.size(); ++i) {
        if (compressed_data[i] == 255 && i + 2 < compressed_data.size()) {
            // Escape marker - next is byte value, then run length
            uint8_t byte_val = compressed_data[++i];
            uint8_t run_length = compressed_data[++i];
            
            // Repeat the byte run_length times
            for (uint8_t j = 0; j < run_length; ++j) {
                decompressed.push_back(byte_val);
            }
        } else {
            // Regular byte
            decompressed.push_back(compressed_data[i]);
        }
    }
    
    return decompressed;
}

std::vector<uint8_t> MemoryCompactionSystem::decompress_zstd(const std::vector<uint8_t>& compressed_data) {
    // Use LZ4 decompression as fallback for ZSTD
    return decompress_lz4(compressed_data);
}

std::vector<uint8_t> MemoryCompactionSystem::decompress_gzip(const std::vector<uint8_t>& compressed_data) {
    // Use LZ4 decompression as fallback for gzip
    return decompress_lz4(compressed_data);
}

std::vector<uint8_t> MemoryCompactionSystem::decompress_brotli(const std::vector<uint8_t>& compressed_data) {
    // Use LZ4 decompression as fallback for Brotli
    return decompress_lz4(compressed_data);
}

// Compaction strategies

std::vector<uint8_t> MemoryCompactionSystem::apply_light_compaction(const std::vector<ConversationEntry>& entries) {
    // Light compaction: Basic deduplication
    auto deduplicated = deduplicate_entries(entries);
    return serialize_entries(deduplicated);
}

std::vector<uint8_t> MemoryCompactionSystem::apply_medium_compaction(const std::vector<ConversationEntry>& entries) {
    // Medium compaction: Remove redundant information, keep key points
    std::vector<ConversationEntry> compacted;

    for (const auto& entry : entries) {
        ConversationEntry compact_entry = entry;

        // Truncate long responses (keep first 500 chars)
        if (compact_entry.hrm_response.length() > 500) {
            compact_entry.hrm_response = compact_entry.hrm_response.substr(0, 500) + "...";
        }

        // Extract topics and entities for better compression
        compact_entry.topics = extract_topics(entry);
        compact_entry.entities = extract_entities(entry);

        compacted.push_back(compact_entry);
    }

    return serialize_entries(compacted);
}

std::vector<uint8_t> MemoryCompactionSystem::apply_heavy_compaction(const std::vector<ConversationEntry>& entries) {
    // Heavy compaction: Generate summaries, remove details
    std::vector<ConversationEntry> compacted;

    // Group entries by topic and create summaries
    std::unordered_map<std::string, std::vector<ConversationEntry>> topic_groups;

    for (const auto& entry : entries) {
        std::string main_topic = entry.topics.empty() ? "general" : entry.topics[0];
        topic_groups[main_topic].push_back(entry);
    }

    for (const auto& pair : topic_groups) {
        ConversationEntry summary_entry;
        summary_entry.id = "summary_" + pair.first;
        summary_entry.timestamp = std::chrono::system_clock::now();
        summary_entry.user_message = "Topic: " + pair.first;
        summary_entry.hrm_response = generate_summary(pair.second);
        summary_entry.confidence_score = 0.8; // Summary confidence
        summary_entry.topics = {pair.first};
        summary_entry.entities = {}; // Could extract from all entries

        compacted.push_back(summary_entry);
    }

    return serialize_entries(compacted);
}

std::vector<uint8_t> MemoryCompactionSystem::apply_extreme_compaction(const std::vector<ConversationEntry>& entries) {
    // Extreme compaction: Lossy compression, keep only essential information
    std::string summary = "Compacted " + std::to_string(entries.size()) + " conversations. ";
    summary += "Topics covered: ";

    std::set<std::string> all_topics_set;
    for (const auto& entry : entries) {
        for (const auto& topic : entry.topics) {
            all_topics_set.insert(topic);
        }
    }

    for (const auto& topic : all_topics_set) {
        summary += topic + ", ";
    }

    // Create a single summary entry
    ConversationEntry summary_entry;
    summary_entry.id = "extreme_compaction_" + generate_compaction_id();
    summary_entry.timestamp = std::chrono::system_clock::now();
    summary_entry.user_message = "Multiple conversations";
    summary_entry.hrm_response = summary;
    summary_entry.confidence_score = 0.5; // Low confidence due to lossy compression

    return serialize_entries({summary_entry});
}

// Helper methods

std::vector<uint8_t> MemoryCompactionSystem::serialize_entries(const std::vector<ConversationEntry>& entries) {
    std::stringstream ss;

    for (const auto& entry : entries) {
        ss << "ENTRY_START\n";
        ss << "ID:" << entry.id << "\n";
        ss << "TIMESTAMP:" << std::chrono::duration_cast<std::chrono::seconds>(
            entry.timestamp.time_since_epoch()).count() << "\n";
        ss << "USER:" << entry.user_message << "\n";
        ss << "HRM:" << entry.hrm_response << "\n";
        ss << "CONFIDENCE:" << entry.confidence_score << "\n";
        ss << "TOPICS:"; for (const auto& topic : entry.topics) ss << topic << ","; ss << "\n";
        ss << "ENTITIES:"; for (const auto& entity : entry.entities) ss << entity << ","; ss << "\n";
        ss << "SIZE:" << entry.original_size_bytes << "\n";
        ss << "ENTRY_END\n";
    }

    std::string data = ss.str();
    return std::vector<uint8_t>(data.begin(), data.end());
}

std::vector<ConversationEntry> MemoryCompactionSystem::deserialize_entries(const std::vector<uint8_t>& data) {
    std::vector<ConversationEntry> entries;
    std::string str_data(data.begin(), data.end());
    std::stringstream ss(str_data);
    std::string line;

    ConversationEntry current_entry;

    while (std::getline(ss, line)) {
        if (line == "ENTRY_START") {
            current_entry = ConversationEntry{};
        } else if (line == "ENTRY_END") {
            if (!current_entry.id.empty()) {
                entries.push_back(current_entry);
            }
        } else if (line.substr(0, 3) == "ID:") {
            current_entry.id = line.substr(3);
        } else if (line.substr(0, 10) == "TIMESTAMP:") {
            long long timestamp = std::stoll(line.substr(10));
            current_entry.timestamp = std::chrono::system_clock::time_point(
                std::chrono::seconds(timestamp));
        } else if (line.substr(0, 5) == "USER:") {
            current_entry.user_message = line.substr(5);
        } else if (line.substr(0, 4) == "HRM:") {
            current_entry.hrm_response = line.substr(4);
        } else if (line.substr(0, 11) == "CONFIDENCE:") {
            current_entry.confidence_score = std::stod(line.substr(11));
        } else if (line.substr(0, 7) == "TOPICS:") {
            std::string topics_str = line.substr(7);
            std::stringstream topics_ss(topics_str);
            std::string topic;
            while (std::getline(topics_ss, topic, ',')) {
                if (!topic.empty()) {
                    current_entry.topics.push_back(topic);
                }
            }
        } else if (line.substr(0, 9) == "ENTITIES:") {
            std::string entities_str = line.substr(9);
            std::stringstream entities_ss(entities_str);
            std::string entity;
            while (std::getline(entities_ss, entity, ',')) {
                if (!entity.empty()) {
                    current_entry.entities.push_back(entity);
                }
            }
        } else if (line.substr(0, 5) == "SIZE:") {
            current_entry.original_size_bytes = std::stoul(line.substr(5));
        }
    }

    return entries;
}

std::string MemoryCompactionSystem::generate_compaction_id() const {
    static std::atomic<uint64_t> counter{0};
    std::stringstream ss;
    ss << "compact_" << std::chrono::system_clock::now().time_since_epoch().count()
       << "_" << counter++;
    return ss.str();
}

std::string MemoryCompactionSystem::get_compaction_file_path(const std::string& compaction_id) const {
    return config_.compaction_directory + "/" + compaction_id + ".compact";
}

bool MemoryCompactionSystem::save_compaction_to_file(const CompactedMemory& compaction) {
    try {
        std::string file_path = get_compaction_file_path(compaction.compaction_id);
        std::ofstream file(file_path, std::ios::binary);

        if (!file.is_open()) {
            return false;
        }

        // Write metadata
        file << compaction.compaction_id << "\n";
        file << std::chrono::duration_cast<std::chrono::seconds>(
            compaction.created_time.time_since_epoch()).count() << "\n";
        file << static_cast<int>(compaction.level) << "\n";
        file << static_cast<int>(compaction.algorithm) << "\n";
        file << compaction.original_size_bytes << "\n";
        file << compaction.compressed_size_bytes << "\n";
        file << compaction.compression_ratio << "\n";

        // Write metadata map
        file << compaction.metadata.size() << "\n";
        for (const auto& pair : compaction.metadata) {
            file << pair.first << "\n" << pair.second << "\n";
        }

        // Write entry IDs
        file << compaction.entry_ids.size() << "\n";
        for (const auto& id : compaction.entry_ids) {
            file << id << "\n";
        }

        // Write compressed data
        file.write(reinterpret_cast<const char*>(compaction.compressed_data.data()),
                  compaction.compressed_data.size());

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to save compaction: " << e.what() << std::endl;
        return false;
    }
}

CompactedMemory MemoryCompactionSystem::load_compaction_from_file(const std::string& compaction_id) const {
    CompactedMemory compaction;

    try {
        std::string file_path = get_compaction_file_path(compaction_id);
        std::ifstream file(file_path, std::ios::binary);

        if (!file.is_open()) {
            return compaction; // Empty result
        }

        // Read metadata
        std::getline(file, compaction.compaction_id);
        std::string timestamp_str;
        std::getline(file, timestamp_str);
        long long timestamp = std::stoll(timestamp_str);
        compaction.created_time = std::chrono::system_clock::time_point(std::chrono::seconds(timestamp));

        std::string level_str, algorithm_str, orig_size_str, comp_size_str, ratio_str;
        std::getline(file, level_str);
        std::getline(file, algorithm_str);
        std::getline(file, orig_size_str);
        std::getline(file, comp_size_str);
        std::getline(file, ratio_str);

        compaction.level = static_cast<MemoryCompactionLevel>(std::stoi(level_str));
        compaction.algorithm = static_cast<CompressionAlgorithm>(std::stoi(algorithm_str));
        compaction.original_size_bytes = std::stoul(orig_size_str);
        compaction.compressed_size_bytes = std::stoul(comp_size_str);
        compaction.compression_ratio = std::stod(ratio_str);

        // Read metadata map
        std::string metadata_size_str;
        std::getline(file, metadata_size_str);
        size_t metadata_size = std::stoul(metadata_size_str);

        for (size_t i = 0; i < metadata_size; ++i) {
            std::string key, value;
            std::getline(file, key);
            std::getline(file, value);
            compaction.metadata[key] = value;
        }

        // Read entry IDs
        std::string entry_count_str;
        std::getline(file, entry_count_str);
        size_t entry_count = std::stoul(entry_count_str);

        for (size_t i = 0; i < entry_count; ++i) {
            std::string entry_id;
            std::getline(file, entry_id);
            compaction.entry_ids.push_back(entry_id);
        }

        // Read compressed data
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(-static_cast<long long>(compaction.compressed_size_bytes), std::ios::end);

        compaction.compressed_data.resize(compaction.compressed_size_bytes);
        file.read(reinterpret_cast<char*>(compaction.compressed_data.data()),
                 compaction.compressed_size_bytes);

    } catch (const std::exception& e) {
        std::cerr << "Failed to load compaction: " << e.what() << std::endl;
        compaction = CompactedMemory{}; // Reset to empty
    }

    return compaction;
}

CompactedMemory MemoryCompactionSystem::download_compaction_from_cloud(const std::string& compaction_id) {
    CompactedMemory compaction;

    if (!cloud_storage_manager_) {
        return compaction;
    }

    try {
        auto download_result = cloud_storage_manager_->download_compacted_memory(compaction_id, config_.default_cloud_provider);

        if (download_result.success) {
            // Deserialize the compacted memory from downloaded data using the same format as file storage
            std::stringstream ss(std::string(download_result.data.begin(), download_result.data.end()));

            // Read metadata
            std::getline(ss, compaction.compaction_id);
            std::string timestamp_str;
            std::getline(ss, timestamp_str);
            long long timestamp = std::stoll(timestamp_str);
            compaction.created_time = std::chrono::system_clock::time_point(std::chrono::seconds(timestamp));

            std::string level_str, algorithm_str, orig_size_str, comp_size_str, ratio_str;
            std::getline(ss, level_str);
            std::getline(ss, algorithm_str);
            std::getline(ss, orig_size_str);
            std::getline(ss, comp_size_str);
            std::getline(ss, ratio_str);

            compaction.level = static_cast<MemoryCompactionLevel>(std::stoi(level_str));
            compaction.algorithm = static_cast<CompressionAlgorithm>(std::stoi(algorithm_str));
            compaction.original_size_bytes = std::stoul(orig_size_str);
            compaction.compressed_size_bytes = std::stoul(comp_size_str);
            compaction.compression_ratio = std::stod(ratio_str);

            // Read metadata map
            std::string metadata_size_str;
            std::getline(ss, metadata_size_str);
            size_t metadata_size = std::stoul(metadata_size_str);
            for (size_t i = 0; i < metadata_size; ++i) {
                std::string key, value;
                std::getline(ss, key);
                std::getline(ss, value);
                compaction.metadata[key] = value;
            }

            // Read entry IDs
            std::string entry_count_str;
            std::getline(ss, entry_count_str);
            size_t entry_count = std::stoul(entry_count_str);
            compaction.entry_ids.resize(entry_count);
            for (size_t i = 0; i < entry_count; ++i) {
                std::getline(ss, compaction.entry_ids[i]);
            }

            // Read compressed data size and data
            std::string data_size_str;
            std::getline(ss, data_size_str);
            size_t data_size = std::stoul(data_size_str);
            compaction.compressed_data.resize(data_size);
            ss.read(reinterpret_cast<char*>(compaction.compressed_data.data()), data_size);

            std::cout << "Downloaded compaction " << compaction_id << " from cloud storage" << std::endl;
        } else {
            std::cerr << "Failed to download compaction " << compaction_id << " from cloud: " << download_result.error_message << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error downloading compaction from cloud: " << e.what() << std::endl;
    }

    return compaction;
}

bool MemoryCompactionSystem::upload_compaction_to_cloud(const CompactedMemory& compaction) {
    if (!cloud_storage_manager_) {
        return false;
    }

    try {
        // Serialize the compacted memory using the same format as file storage
        std::stringstream ss;

        // Write metadata
        ss << compaction.compaction_id << "\n";
        ss << std::chrono::duration_cast<std::chrono::seconds>(
            compaction.created_time.time_since_epoch()).count() << "\n";
        ss << static_cast<int>(compaction.level) << "\n";
        ss << static_cast<int>(compaction.algorithm) << "\n";
        ss << compaction.original_size_bytes << "\n";
        ss << compaction.compressed_size_bytes << "\n";
        ss << compaction.compression_ratio << "\n";

        // Write metadata map
        ss << compaction.metadata.size() << "\n";
        for (const auto& pair : compaction.metadata) {
            ss << pair.first << "\n" << pair.second << "\n";
        }

        // Write entry IDs
        ss << compaction.entry_ids.size() << "\n";
        for (const auto& id : compaction.entry_ids) {
            ss << id << "\n";
        }

        // Write compressed data
        ss << compaction.compressed_data.size() << "\n";
        ss.write(reinterpret_cast<const char*>(compaction.compressed_data.data()), compaction.compressed_data.size());

        std::string data_str = ss.str();
        std::vector<uint8_t> data(data_str.begin(), data_str.end());

        auto upload_result = cloud_storage_manager_->upload_compacted_memory(compaction.compaction_id, data, config_.default_cloud_provider);

        if (upload_result.success) {
            std::cout << "Uploaded compaction " << compaction.compaction_id << " to cloud storage" << std::endl;
            return true;
        } else {
            std::cerr << "Failed to upload compaction " << compaction.compaction_id << " to cloud: " << upload_result.error_message << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error uploading compaction to cloud: " << e.what() << std::endl;
        return false;
    }
}

std::vector<std::string> MemoryCompactionSystem::extract_topics(const ConversationEntry& entry) {
    // Simple topic extraction based on keywords
    std::vector<std::string> topics;
    std::string text = entry.user_message + " " + entry.hrm_response;

    std::transform(text.begin(), text.end(), text.begin(), ::tolower);

    if (text.find("quantum") != std::string::npos || text.find("physics") != std::string::npos) {
        topics.push_back("physics");
    }
    if (text.find("ai") != std::string::npos || text.find("artificial") != std::string::npos) {
        topics.push_back("artificial_intelligence");
    }
    if (text.find("code") != std::string::npos || text.find("programming") != std::string::npos) {
        topics.push_back("programming");
    }

    return topics;
}

std::vector<std::string> MemoryCompactionSystem::extract_entities(const ConversationEntry& entry) {
    // Simple entity extraction (could be enhanced with NLP)
    std::vector<std::string> entities;
    std::string text = entry.user_message + " " + entry.hrm_response;

    // Look for capitalized words (potential names)
    std::regex name_regex("\\b[A-Z][a-z]+\\b");
    std::smatch matches;
    std::string::const_iterator search_start(text.cbegin());

    while (std::regex_search(search_start, text.cend(), matches, name_regex)) {
        entities.push_back(matches[0]);
        search_start = matches.suffix().first;
    }

    return entities;
}

std::string MemoryCompactionSystem::generate_summary(const std::vector<ConversationEntry>& entries) {
    if (entries.empty()) return "No conversations to summarize";

    std::stringstream ss;
    ss << "Summary of " << entries.size() << " conversations. ";

    // Count topics
    std::unordered_map<std::string, int> topic_counts;
    for (const auto& entry : entries) {
        for (const auto& topic : entry.topics) {
            topic_counts[topic]++;
        }
    }

    if (!topic_counts.empty()) {
        ss << "Main topics: ";
        for (const auto& pair : topic_counts) {
            if (pair.second > entries.size() / 3) { // Mention if > 1/3 of conversations
                ss << pair.first << " (" << pair.second << " conversations), ";
            }
        }
    }

    ss << "Average confidence: ";
    double avg_confidence = 0.0;
    for (const auto& entry : entries) {
        avg_confidence += entry.confidence_score;
    }
    avg_confidence /= entries.size();
    ss << std::fixed << std::setprecision(2) << avg_confidence * 100 << "%";

    return ss.str();
}

std::vector<ConversationEntry> MemoryCompactionSystem::deduplicate_entries(const std::vector<ConversationEntry>& entries) {
    std::vector<ConversationEntry> deduplicated;
    std::unordered_set<std::string> seen_messages;

    for (const auto& entry : entries) {
        std::string key = entry.user_message + "|" + entry.hrm_response;
        if (seen_messages.find(key) == seen_messages.end()) {
            deduplicated.push_back(entry);
            seen_messages.insert(key);
        }
    }

    return deduplicated;
}

size_t MemoryCompactionSystem::calculate_memory_usage(const std::vector<ConversationEntry>& entries) const {
    size_t total = 0;
    for (const auto& entry : entries) {
        total += entry.original_size_bytes;
    }
    return total;
}

std::vector<ConversationEntry> MemoryCompactionSystem::select_entries_for_compaction() const {
    // Select older entries that are not in the recent window
    auto cutoff_time = std::chrono::system_clock::now() - config_.recent_conversation_window;

    std::vector<ConversationEntry> candidates;
    for (const auto& entry : recent_conversations_) {
        if (entry.timestamp < cutoff_time) {
            candidates.push_back(entry);
        }
    }

    // Limit to prevent excessive compaction
    if (candidates.size() > 1000) {
        candidates.resize(1000);
    }

    return candidates;
}

std::vector<ConversationEntry> MemoryCompactionSystem::prioritize_entries_for_retention() const {
    // Keep recent and high-confidence conversations
    auto cutoff_time = std::chrono::system_clock::now() - config_.recent_conversation_window;

    std::vector<ConversationEntry> to_retain;
    for (const auto& entry : recent_conversations_) {
        if (entry.timestamp > cutoff_time || entry.confidence_score > 0.8) {
            to_retain.push_back(entry);
        }
    }

    return to_retain;
}

void MemoryCompactionSystem::auto_compaction_loop() {
    while (auto_compaction_running_) {
        std::this_thread::sleep_for(config_.compaction_interval);

        if (should_trigger_compaction()) {
            perform_auto_compaction();
        }
    }
}

bool MemoryCompactionSystem::should_trigger_compaction() const {
    if (!config_.auto_compaction_enabled) return false;

    size_t current_usage = get_current_memory_usage();
    return current_usage > (config_.max_memory_before_compaction_mb * 1024 * 1024);
}

void MemoryCompactionSystem::perform_auto_compaction() {
    std::cout << "Performing automatic memory compaction..." << std::endl;

    auto result = compact_memory_auto();

    if (result.success) {
        std::cout << "Auto-compaction completed: " << result.compression_ratio << "x compression ratio" << std::endl;

        // Clean up compacted entries from recent conversations
        auto entries_to_remove = select_entries_for_compaction();
        for (const auto& entry : entries_to_remove) {
            auto it = std::find_if(recent_conversations_.begin(), recent_conversations_.end(),
                                 [&entry](const ConversationEntry& e) { return e.id == entry.id; });
            if (it != recent_conversations_.end()) {
                recent_conversations_.erase(it);
            }
        }
    } else {
        std::cout << "Auto-compaction failed: " << result.error_message << std::endl;
    }
}