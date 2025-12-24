#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include <mutex>
#include <chrono>

/**
 * RAM-based cache for character sequences to reduce redundant processing
 * Used for CPU/RAM offloading optimization
 */
class CharacterSequenceCache {
public:
    struct CacheEntry {
        std::vector<uint32_t> encoded_sequence;
        std::chrono::system_clock::time_point last_access;
        size_t access_count;
    };

    CharacterSequenceCache(size_t max_entries = 1000, size_t max_memory_mb = 100);
    ~CharacterSequenceCache() = default;

    // Cache operations
    bool get(const std::string& key, std::vector<uint32_t>& sequence);
    void put(const std::string& key, const std::vector<uint32_t>& sequence);
    void clear();
    size_t size() const;
    size_t memory_usage_mb() const;

    // Cache management
    void set_max_entries(size_t max_entries);
    void set_max_memory_mb(size_t max_mb);
    void cleanup_expired_entries(std::chrono::seconds max_age);

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, CacheEntry> cache_;
    size_t max_entries_;
    size_t max_memory_mb_;
    size_t current_memory_bytes_;

    void evict_lru();
    size_t estimate_memory_usage(const std::vector<uint32_t>& sequence) const;
};