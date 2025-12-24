#pragma once

#include <unordered_map>
#include <vector>
#include <memory>
#include <functional>
#include <chrono>
#include <cstdint>

namespace Nyx {

// Cache entry for quantized models/components
struct CacheEntry {
    std::string key;
    std::vector<uint8_t> data;
    size_t size_bytes;
    std::chrono::system_clock::time_point last_access;
    std::chrono::system_clock::time_point created;
    int access_count;
    float priority_score;
    std::string cache_level;  // "L1", "L2", "RAM", "GPU"
};

// Intelligent cache manager
class IntelligentCacheManager {
public:
    IntelligentCacheManager(size_t max_cache_size_mb = 1024);

    // Cache operations
    bool store(const std::string& key, const std::vector<uint8_t>& data,
              const std::string& level = "RAM");
    std::vector<uint8_t> retrieve(const std::string& key);
    bool contains(const std::string& key) const;
    void remove(const std::string& key);

    // Multi-level cache management
    void set_cache_level_size(const std::string& level, size_t max_size_mb);
    void promote_to_higher_level(const std::string& key);
    void demote_to_lower_level(const std::string& key);

    // Intelligent prefetching
    void prefetch_likely_needed(const std::vector<std::string>& keys);
    void analyze_access_patterns();
    std::vector<std::string> predict_future_accesses();

    // Cache optimization
    void optimize_cache_usage();
    void compress_cache_entries();
    void cleanup_expired_entries();

    // Performance monitoring
    CacheStatistics get_statistics() const;
    std::vector<CacheEntry> get_hot_entries() const;

private:
    // Multi-level cache storage
    std::unordered_map<std::string, CacheEntry> l1_cache_;  // Fastest, smallest
    std::unordered_map<std::string, CacheEntry> l2_cache_;  // Medium speed/size
    std::unordered_map<std::string, CacheEntry> ram_cache_; // Large RAM cache
    std::unordered_map<std::string, CacheEntry> gpu_cache_; // GPU memory cache

    // Cache size limits
    size_t max_l1_size_mb_ = 64;   // 64MB L1 cache
    size_t max_l2_size_mb_ = 256;  // 256MB L2 cache
    size_t max_ram_size_mb_ = 1024; // 1GB RAM cache
    size_t max_gpu_size_mb_ = 512;  // 512MB GPU cache

    // Current cache sizes
    size_t current_l1_size_ = 0;
    size_t current_l2_size_ = 0;
    size_t current_ram_size_ = 0;
    size_t current_gpu_size_ = 0;

    // Cache replacement policies
    void apply_lru_eviction(std::unordered_map<std::string, CacheEntry>& cache,
                           size_t& current_size, size_t max_size);
    void apply_lfu_eviction(std::unordered_map<std::string, CacheEntry>& cache,
                           size_t& current_size, size_t max_size);
    void apply_adaptive_eviction(std::unordered_map<std::string, CacheEntry>& cache,
                                size_t& current_size, size_t max_size);

    // Access pattern analysis
    std::vector<std::string> access_history_;
    std::unordered_map<std::string, int> access_frequency_;
    void update_access_patterns(const std::string& key);

    // Prediction algorithms
    std::vector<std::string> markov_chain_prediction();
    std::vector<std::string> frequency_based_prediction();
    std::vector<std::string> temporal_pattern_prediction();
};

// Cache-aware quantization engine
class CacheAwareQuantizationEngine {
public:
    CacheAwareQuantizationEngine(std::shared_ptr<IntelligentCacheManager> cache_manager);

    // Cache-aware quantization
    QuantizedModel quantize_with_caching(const Model& model, const QuantizationConfig& config);

    // Intelligent model loading with caching
    QuantizedModel load_cached_quantized_model(const std::string& model_key,
                                             const QuantizationConfig& config);

    // Cache prefetching for quantized operations
    void prefetch_quantized_components(const QuantizationConfig& config,
                                     const std::vector<std::string>& component_keys);

    // Memory-efficient quantization with caching
    void quantize_streaming_with_cache(const Model& model, const QuantizationConfig& config,
                                     std::function<void(const QuantizedModel&)> callback);

private:
    std::shared_ptr<IntelligentCacheManager> cache_manager_;

    // Cache key generation
    std::string generate_cache_key(const Model& model, const QuantizationConfig& config);
    std::string generate_component_key(const std::string& component_name,
                                     const QuantizationConfig& config);

    // Cache-aware quantization strategies
    QuantizedModel quantize_with_memory_efficiency(const Model& model,
                                                  const QuantizationConfig& config);
    void optimize_cache_layout(const QuantizedModel& model);
};

// Performance statistics
struct CacheStatistics {
    size_t total_entries;
    size_t total_size_bytes;
    double hit_rate;
    double miss_rate;
    std::chrono::milliseconds avg_access_time;
    size_t evictions_count;
    size_t prefetches_count;
    std::unordered_map<std::string, size_t> level_sizes;
    std::unordered_map<std::string, double> level_hit_rates;
};

// Cache compression utilities
class CacheCompression {
public:
    static std::vector<uint8_t> compress_data(const std::vector<uint8_t>& data,
                                            CompressionType type = CompressionType::LZ4);
    static std::vector<uint8_t> decompress_data(const std::vector<uint8_t>& compressed_data,
                                              CompressionType type = CompressionType::LZ4);

    static size_t estimate_compressed_size(const std::vector<uint8_t>& data,
                                         CompressionType type = CompressionType::LZ4);

private:
    enum class CompressionType { LZ4, ZSTD, NONE };
};

// Forward declarations for missing types
struct Model;
struct QuantizedModel;
struct QuantizationConfig;
struct HardwareCapabilities;
struct TaskRequirements;
class AdaptiveQuantizationManager;
class MetaQuantizationLearner;

} // namespace Nyx