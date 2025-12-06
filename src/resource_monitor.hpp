#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include <functional>

struct ResourceUsage {
    // CPU metrics
    double cpu_usage_percent;
    double cpu_temperature_celsius;
    std::vector<double> cpu_core_usage;

    // Memory metrics
    uint64_t total_memory_bytes;
    uint64_t available_memory_bytes;
    uint64_t used_memory_bytes;
    double memory_usage_percent;

    // Disk metrics
    uint64_t total_disk_bytes;
    uint64_t available_disk_bytes;
    uint64_t used_disk_bytes;
    double disk_usage_percent;
    double disk_read_speed_mbps;
    double disk_write_speed_mbps;

    // Network metrics
    double network_download_speed_mbps;
    double network_upload_speed_mbps;
    uint64_t network_bytes_received;
    uint64_t network_bytes_sent;

    // System metrics
    double system_load_average;
    uint32_t process_count;
    std::chrono::system_clock::time_point timestamp;
};

struct ResourceThresholds {
    double max_cpu_usage_percent;
    double max_memory_usage_percent;
    uint64_t min_available_memory_bytes;
    double max_disk_usage_percent;
    uint64_t min_available_disk_bytes;

    // Action thresholds
    double warning_cpu_threshold;
    double warning_memory_threshold;
    double critical_cpu_threshold;
    double critical_memory_threshold;
};

enum class ResourceAlertLevel {
    NORMAL,
    WARNING,
    CRITICAL,
    EMERGENCY
};

struct ResourceAlert {
    ResourceAlertLevel level;
    std::string resource_type;
    std::string message;
    double current_value;
    double threshold_value;
    std::chrono::system_clock::time_point timestamp;
};

class ResourceMonitor {
public:
    ResourceMonitor();
    ~ResourceMonitor();

    // Monitoring control
    void start_monitoring(std::chrono::milliseconds interval = std::chrono::milliseconds(1000));
    void stop_monitoring();
    bool is_monitoring() const;

    // Resource queries
    ResourceUsage get_current_usage() const;
    ResourceUsage get_average_usage(std::chrono::seconds window) const;
    ResourceThresholds get_thresholds() const;
    void set_thresholds(const ResourceThresholds& thresholds);

    // Alert system
    std::vector<ResourceAlert> get_active_alerts() const;
    std::vector<ResourceAlert> get_alert_history(std::chrono::seconds window) const;
    void clear_alerts();

    // Resource prediction
    double predict_memory_usage_mb(int seconds_ahead) const;
    double predict_cpu_usage_percent(int seconds_ahead) const;

    // System information
    std::unordered_map<std::string, std::string> get_system_info() const;
    
    // Resource-aware timeout calculations
    std::chrono::milliseconds calculate_adaptive_timeout(std::chrono::milliseconds base_timeout, 
                                                       double complexity_factor = 1.0) const;
    std::chrono::seconds calculate_process_timeout(const std::string& operation_type = "default") const;
    std::chrono::milliseconds calculate_gpu_timeout(size_t operation_size = 1) const;

private:
    // Monitoring thread
    std::thread monitoring_thread_;
    std::atomic<bool> monitoring_active_;
    std::chrono::milliseconds monitoring_interval_;

    // Data storage
    mutable std::mutex data_mutex_;
    std::vector<ResourceUsage> usage_history_;
    ResourceThresholds thresholds_;
    std::vector<ResourceAlert> active_alerts_;
    std::vector<ResourceAlert> alert_history_;

    // Platform-specific monitoring functions
    ResourceUsage collect_linux_resources();
    ResourceUsage collect_windows_resources();
    ResourceUsage collect_macos_resources();

    // Helper functions
    double calculate_cpu_usage();
    ResourceUsage get_memory_info();
    ResourceUsage get_disk_info();
    ResourceUsage get_network_info();
    double get_system_load();
#ifdef _WIN32
    double get_cpu_usage_windows() const;
#endif

    // Alert generation
    void check_thresholds(const ResourceUsage& usage);
    void generate_alert(ResourceAlertLevel level, const std::string& resource_type,
                       const std::string& message, double current_value, double threshold_value);

    // History management
    void add_usage_sample(const ResourceUsage& usage);
    void cleanup_old_data();
};