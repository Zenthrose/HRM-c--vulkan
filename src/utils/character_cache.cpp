#include "character_cache.hpp"
#include <algorithm>

CharacterSequenceCache::CharacterSequenceCache(size_t max_entries, size_t max_memory_mb)
    : max_entries_(max_entries), max_memory_mb_(max_memory_mb), current_memory_bytes_(0) {}

bool CharacterSequenceCache::get(const std::string& key, std::vector<uint32_t>& sequence) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        it->second.last_access = std::chrono::system_clock::now();
        it->second.access_count++;
        sequence = it->second.encoded_sequence;
        return true;
    }
    return false;
}

void CharacterSequenceCache::put(const std::string& key, const std::vector<uint32_t>& sequence) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if we need to evict
    size_t new_memory = estimate_memory_usage(sequence);
    while (cache_.size() >= max_entries_ ||
           (current_memory_bytes_ + new_memory) > (max_memory_mb_ * 1024 * 1024)) {
        evict_lru();
    }

    // Add new entry
    CacheEntry entry{sequence, std::chrono::system_clock::now(), 1};
    cache_[key] = entry;
    current_memory_bytes_ += new_memory;
}

void CharacterSequenceCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    current_memory_bytes_ = 0;
}

size_t CharacterSequenceCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.size();
}

size_t CharacterSequenceCache::memory_usage_mb() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_memory_bytes_ / (1024 * 1024);
}

void CharacterSequenceCache::set_max_entries(size_t max_entries) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_entries_ = max_entries;
    while (cache_.size() > max_entries_) {
        evict_lru();
    }
}

void CharacterSequenceCache::set_max_memory_mb(size_t max_mb) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_memory_mb_ = max_mb;
    while (current_memory_bytes_ > max_memory_mb_ * 1024 * 1024) {
        evict_lru();
    }
}

void CharacterSequenceCache::cleanup_expired_entries(std::chrono::seconds max_age) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto now = std::chrono::system_clock::now();
    for (auto it = cache_.begin(); it != cache_.end();) {
        if (now - it->second.last_access > max_age) {
            current_memory_bytes_ -= estimate_memory_usage(it->second.encoded_sequence);
            it = cache_.erase(it);
        } else {
            ++it;
        }
    }
}

void CharacterSequenceCache::evict_lru() {
    if (cache_.empty()) return;

    auto lru_it = std::min_element(cache_.begin(), cache_.end(),
        [](const auto& a, const auto& b) {
            return a.second.last_access < b.second.last_access;
        });

    current_memory_bytes_ -= estimate_memory_usage(lru_it->second.encoded_sequence);
    cache_.erase(lru_it);
}

size_t CharacterSequenceCache::estimate_memory_usage(const std::vector<uint32_t>& sequence) const {
    // Estimate memory: vector overhead + data
    return sizeof(std::vector<uint32_t>) + sequence.size() * sizeof(uint32_t);
}