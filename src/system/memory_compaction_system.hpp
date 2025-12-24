#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <memory>
#include <chrono>
#include <filesystem>
#include <thread>
#include <atomic>
#include "cloud_storage_manager.hpp"

namespace fs = std::filesystem;

enum class CompressionAlgorithm {
    LZ4,
    ZSTD,
    GZIP,
    BROTLI,
    NONE
};

enum class MemoryCompactionLevel {
    LIGHT,      // Basic deduplication
    MEDIUM,     // Semantic compression
    HEAVY,      // Aggressive summarization
    EXTREME     // Lossy compression
};

struct ConversationEntry {
    std::string id;
    std::chrono::system_clock::time_point timestamp;
    std::string user_message;
    std::string hrm_response;
    double confidence_score;
    std::vector<std::string> topics;
    std::vector<std::string> entities;
    size_t original_size_bytes;
};

struct CompactedMemory {
    std::string compaction_id;
    std::chrono::system_clock::time_point created_time;
    MemoryCompactionLevel level;
    CompressionAlgorithm algorithm;
    std::vector<uint8_t> compressed_data;
    std::unordered_map<std::string, std::string> metadata;
    size_t original_size_bytes;
    size_t compressed_size_bytes;
    double compression_ratio;
    std::vector<std::string> entry_ids; // IDs of entries in this compaction
};

struct MemoryCompactionResult {
    bool success;
    std::string compaction_id;
    size_t original_size;
    size_t compressed_size;
    double compression_ratio;
    std::chrono::milliseconds processing_time;
    std::vector<std::string> warnings;
    std::string error_message;
};

struct MemoryCompactionConfig {
    MemoryCompactionLevel default_level;
    CompressionAlgorithm preferred_algorithm;
    size_t max_memory_before_compaction_mb;
    size_t target_memory_after_compaction_mb;
    bool auto_compaction_enabled;
    std::chrono::hours compaction_interval;
    bool preserve_recent_conversations;
    std::chrono::hours recent_conversation_window;
    std::string compaction_directory;
    std::shared_ptr<CloudStorageManager> cloud_storage_manager;
    CloudProvider default_cloud_provider;
};

class MemoryCompactionSystem {
public:
    MemoryCompactionSystem(const MemoryCompactionConfig& config);
    ~MemoryCompactionSystem();

    // Memory management
    MemoryCompactionResult compact_memory(const std::vector<ConversationEntry>& entries,
                                        MemoryCompactionLevel level = MemoryCompactionLevel::MEDIUM);
    MemoryCompactionResult compact_memory_auto();

    // Memory retrieval and decompression
    std::vector<ConversationEntry> decompress_memory(const std::string& compaction_id);
    std::vector<ConversationEntry> retrieve_recent_conversations(std::chrono::hours window);

    // Compaction management
    std::vector<std::string> list_compactions() const;
    CompactedMemory get_compaction_info(const std::string& compaction_id) const;
    bool delete_compaction(const std::string& compaction_id);
    void cleanup_old_compactions(std::chrono::hours max_age_hours);

    // Memory monitoring
    size_t get_current_memory_usage() const;
    size_t get_compacted_memory_size() const;
    double get_average_compression_ratio() const;
    std::unordered_map<std::string, size_t> get_memory_stats() const;

    // Configuration
    void update_config(const MemoryCompactionConfig& new_config);
    MemoryCompactionConfig get_config() const;

    // Auto-compaction
    void enable_auto_compaction(bool enable);
    bool is_auto_compaction_enabled() const;
    void trigger_auto_compaction();

    // Automatic cleanup
    void perform_memory_cleanup();
    void set_cleanup_thresholds(size_t max_local_compactions, std::chrono::hours max_age_hours);

private:
    MemoryCompactionConfig config_;
    std::unordered_map<std::string, CompactedMemory> compacted_memories_;
    std::vector<ConversationEntry> recent_conversations_;
    std::thread auto_compaction_thread_;
    std::atomic<bool> auto_compaction_running_;
    std::shared_ptr<CloudStorageManager> cloud_storage_manager_;

    // Compression algorithms
    std::vector<uint8_t> compress_lz4(const std::vector<uint8_t>& data);
    std::vector<uint8_t> compress_zstd(const std::vector<uint8_t>& data);
    std::vector<uint8_t> compress_gzip(const std::vector<uint8_t>& data);
    std::vector<uint8_t> compress_brotli(const std::vector<uint8_t>& data);

    std::vector<uint8_t> decompress_lz4(const std::vector<uint8_t>& compressed_data);
    std::vector<uint8_t> decompress_zstd(const std::vector<uint8_t>& compressed_data);
    std::vector<uint8_t> decompress_gzip(const std::vector<uint8_t>& compressed_data);
    std::vector<uint8_t> decompress_brotli(const std::vector<uint8_t>& compressed_data);

    // Compaction strategies
    std::vector<uint8_t> apply_light_compaction(const std::vector<ConversationEntry>& entries);
    std::vector<uint8_t> apply_medium_compaction(const std::vector<ConversationEntry>& entries);
    std::vector<uint8_t> apply_heavy_compaction(const std::vector<ConversationEntry>& entries);
    std::vector<uint8_t> apply_extreme_compaction(const std::vector<ConversationEntry>& entries);

    // Helper functions
    std::vector<uint8_t> serialize_entries(const std::vector<ConversationEntry>& entries);
    std::vector<ConversationEntry> deserialize_entries(const std::vector<uint8_t>& data);
    std::string generate_compaction_id() const;
    std::string get_compaction_file_path(const std::string& compaction_id) const;
    bool save_compaction_to_file(const CompactedMemory& compaction);
    CompactedMemory load_compaction_from_file(const std::string& compaction_id) const;

    // Cloud storage integration
    CompactedMemory download_compaction_from_cloud(const std::string& compaction_id);
    bool upload_compaction_to_cloud(const CompactedMemory& compaction);

    // Semantic analysis for compression
    std::vector<std::string> extract_topics(const ConversationEntry& entry);
    std::vector<std::string> extract_entities(const ConversationEntry& entry);
    std::string generate_summary(const std::vector<ConversationEntry>& entries);
    std::vector<ConversationEntry> deduplicate_entries(const std::vector<ConversationEntry>& entries);

    // Auto-compaction
    void auto_compaction_loop();
    bool should_trigger_compaction() const;
    void perform_auto_compaction();

    // Memory analysis
    size_t calculate_memory_usage(const std::vector<ConversationEntry>& entries) const;
    std::vector<ConversationEntry> select_entries_for_compaction() const;
    std::vector<ConversationEntry> prioritize_entries_for_retention() const;
};