#include "resource_monitor.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cstring>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <windows.h>
#include <psapi.h>
#include <iphlpapi.h>
#pragma comment(lib, "pdh.lib")
#pragma comment(lib, "iphlpapi.lib")
#pragma comment(lib, "ws2_32.lib")
#else
#include <unistd.h>
#include <sys/sysinfo.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#endif

ResourceMonitor::ResourceMonitor()
    : monitoring_active_(false), monitoring_interval_(std::chrono::milliseconds(1000)) {

    // Set default thresholds
    thresholds_ = {
        80.0,  // max_cpu_usage_percent
        85.0,  // max_memory_usage_percent
        100 * 1024 * 1024,  // min_available_memory_bytes (100MB)
        90.0,  // max_disk_usage_percent
        1024 * 1024 * 1024,  // min_available_disk_bytes (1GB)
        70.0,  // warning_cpu_threshold
        75.0,  // warning_memory_threshold
        90.0,  // critical_cpu_threshold
        90.0   // critical_memory_threshold
    };

    std::cout << "Resource Monitor initialized" << std::endl;
}

ResourceMonitor::~ResourceMonitor() {
    stop_monitoring();
}

void ResourceMonitor::start_monitoring(std::chrono::milliseconds interval) {
    if (monitoring_active_) return;

    monitoring_active_ = true;
    monitoring_interval_ = interval;

    monitoring_thread_ = std::thread([this]() {
        while (monitoring_active_) {
            auto usage = collect_linux_resources(); // Assuming Linux for now
            add_usage_sample(usage);
            check_thresholds(usage);
            cleanup_old_data();

            std::this_thread::sleep_for(monitoring_interval_);
        }
    });

    std::cout << "Resource monitoring started with " << interval.count() << "ms interval" << std::endl;
}

void ResourceMonitor::stop_monitoring() {
    if (!monitoring_active_) return;

    monitoring_active_ = false;
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }

    std::cout << "Resource monitoring stopped" << std::endl;
}

bool ResourceMonitor::is_monitoring() const {
    return monitoring_active_;
}

ResourceUsage ResourceMonitor::get_current_usage() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (usage_history_.empty()) {
        return ResourceUsage{};
    }
    return usage_history_.back();
}

ResourceUsage ResourceMonitor::get_average_usage(std::chrono::seconds window) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (usage_history_.empty()) {
        return ResourceUsage{};
    }

    auto now = std::chrono::system_clock::now();
    auto cutoff = now - window;

    std::vector<ResourceUsage> recent_usage;
    for (const auto& usage : usage_history_) {
        if (usage.timestamp > cutoff) {
            recent_usage.push_back(usage);
        }
    }

    if (recent_usage.empty()) {
        return usage_history_.back();
    }

    // Calculate averages
    ResourceUsage average = recent_usage[0];
    for (size_t i = 1; i < recent_usage.size(); ++i) {
        average.cpu_usage_percent += recent_usage[i].cpu_usage_percent;
        average.memory_usage_percent += recent_usage[i].memory_usage_percent;
        average.disk_usage_percent += recent_usage[i].disk_usage_percent;
        average.network_download_speed_mbps += recent_usage[i].network_download_speed_mbps;
        average.network_upload_speed_mbps += recent_usage[i].network_upload_speed_mbps;
    }

    size_t count = recent_usage.size();
    average.cpu_usage_percent /= count;
    average.memory_usage_percent /= count;
    average.disk_usage_percent /= count;
    average.network_download_speed_mbps /= count;
    average.network_upload_speed_mbps /= count;

    return average;
}

ResourceThresholds ResourceMonitor::get_thresholds() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return thresholds_;
}

void ResourceMonitor::set_thresholds(const ResourceThresholds& thresholds) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    thresholds_ = thresholds;
}

std::vector<ResourceAlert> ResourceMonitor::get_active_alerts() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return active_alerts_;
}

std::vector<ResourceAlert> ResourceMonitor::get_alert_history(std::chrono::seconds window) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto now = std::chrono::system_clock::now();
    auto cutoff = now - window;

    std::vector<ResourceAlert> recent_alerts;
    for (const auto& alert : alert_history_) {
        if (alert.timestamp > cutoff) {
            recent_alerts.push_back(alert);
        }
    }

    return recent_alerts;
}

void ResourceMonitor::clear_alerts() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    active_alerts_.clear();
}

double ResourceMonitor::predict_memory_usage_mb(int seconds_ahead) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (usage_history_.size() < 2) return 0.0;

    // Simple linear regression for prediction
    size_t n = std::min(size_t(10), usage_history_.size());
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;

    for (size_t i = usage_history_.size() - n; i < usage_history_.size(); ++i) {
        double x = i - (usage_history_.size() - n);
        double y = usage_history_[i].used_memory_bytes / (1024.0 * 1024.0); // MB
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
    }

    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    double intercept = (sum_y - slope * sum_x) / n;

    return slope * (n + seconds_ahead) + intercept;
}

double ResourceMonitor::predict_cpu_usage_percent(int seconds_ahead) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (usage_history_.size() < 2) return 0.0;

    // Simple moving average prediction
    size_t n = std::min(size_t(5), usage_history_.size());
    double recent_avg = 0.0;

    for (size_t i = usage_history_.size() - n; i < usage_history_.size(); ++i) {
        recent_avg += usage_history_[i].cpu_usage_percent;
    }

    return recent_avg / n; // Simple prediction: maintain recent average
}

std::unordered_map<std::string, std::string> ResourceMonitor::get_system_info() const {
    std::unordered_map<std::string, std::string> info;

    // Get system information
#ifdef _WIN32
    MEMORYSTATUSEX memStatus;
    memStatus.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memStatus)) {
        info["total_memory_mb"] = std::to_string(memStatus.ullTotalPhys / (1024 * 1024));
        info["free_memory_mb"] = std::to_string(memStatus.ullAvailPhys / (1024 * 1024));
        info["memory_usage_percent"] = std::to_string((1.0 - (double)memStatus.ullAvailPhys / memStatus.ullTotalPhys) * 100.0);
    }
    info["uptime_seconds"] = std::to_string(GetTickCount64() / 1000);
    // CPU load - simplified for Windows
    info["cpu_usage_percent"] = get_cpu_usage_windows();
#else
    struct sysinfo sys_info;
    if (sysinfo(&sys_info) == 0) {
        info["total_memory_mb"] = std::to_string(sys_info.totalram / (1024 * 1024));
        info["free_memory_mb"] = std::to_string(sys_info.freeram / (1024 * 1024));
        info["shared_memory_mb"] = std::to_string(sys_info.sharedram / (1024 * 1024));
        info["buffer_memory_mb"] = std::to_string(sys_info.bufferram / (1024 * 1024));
        info["uptime_seconds"] = std::to_string(sys_info.uptime);
        info["load_average_1min"] = std::to_string(sys_info.loads[0] / 65536.0);
        info["load_average_5min"] = std::to_string(sys_info.loads[1] / 65536.0);
        info["load_average_15min"] = std::to_string(sys_info.loads[2] / 65536.0);
    }
#endif

    // CPU info
#ifdef _WIN32
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    info["cpu_count"] = std::to_string(sysInfo.dwNumberOfProcessors);
    info["cpu_architecture"] = "x64"; // Simplified
#else
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    int cpu_count = 0;
    while (std::getline(cpuinfo, line)) {
        if (line.find("processor") != std::string::npos) {
            cpu_count++;
        }
    }
    info["cpu_count"] = std::to_string(cpu_count);
#endif

    return info;
}

// Private methods

ResourceUsage ResourceMonitor::collect_linux_resources() {
    ResourceUsage usage;
    usage.timestamp = std::chrono::system_clock::now();

    // CPU usage
    usage.cpu_usage_percent = calculate_cpu_usage();

    // Memory info
    auto mem_info = get_memory_info();
    usage.total_memory_bytes = mem_info.total_memory_bytes;
    usage.available_memory_bytes = mem_info.available_memory_bytes;
    usage.used_memory_bytes = mem_info.used_memory_bytes;
    usage.memory_usage_percent = mem_info.memory_usage_percent;

    // Disk info
    auto disk_info = get_disk_info();
    usage.total_disk_bytes = disk_info.total_disk_bytes;
    usage.available_disk_bytes = disk_info.available_disk_bytes;
    usage.used_disk_bytes = disk_info.used_disk_bytes;
    usage.disk_usage_percent = disk_info.disk_usage_percent;

    // Network info
    auto net_info = get_network_info();
    usage.network_download_speed_mbps = net_info.network_download_speed_mbps;
    usage.network_upload_speed_mbps = net_info.network_upload_speed_mbps;
    usage.network_bytes_received = net_info.network_bytes_received;
    usage.network_bytes_sent = net_info.network_bytes_sent;

    // System load
    usage.system_load_average = get_system_load();

    return usage;
}

ResourceUsage ResourceMonitor::collect_windows_resources() {
    // Placeholder for Windows implementation
    return ResourceUsage{};
}

ResourceUsage ResourceMonitor::collect_macos_resources() {
    // Placeholder for macOS implementation
    return ResourceUsage{};
}

double ResourceMonitor::calculate_cpu_usage() {
    static unsigned long long prev_total = 0, prev_idle = 0;

    std::ifstream stat_file("/proc/stat");
    std::string line;
    std::getline(stat_file, line);

    std::istringstream iss(line);
    std::string cpu_label;
    unsigned long long user, nice, system, idle, iowait, irq, softirq, steal;

    iss >> cpu_label >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;

    unsigned long long total = user + nice + system + idle + iowait + irq + softirq + steal;
    unsigned long long total_idle = idle + iowait;

    unsigned long long total_diff = total - prev_total;
    unsigned long long idle_diff = total_idle - prev_idle;

    prev_total = total;
    prev_idle = total_idle;

    if (total_diff == 0) return 0.0;

    return 100.0 * (total_diff - idle_diff) / total_diff;
}

ResourceUsage ResourceMonitor::get_memory_info() {
    ResourceUsage usage;
    std::ifstream meminfo("/proc/meminfo");
    std::string line;

    while (std::getline(meminfo, line)) {
        std::istringstream iss(line);
        std::string key;
        unsigned long long value;
        std::string unit;

        iss >> key >> value >> unit;

        if (key == "MemTotal:") {
            usage.total_memory_bytes = value * 1024; // Convert KB to bytes
        } else if (key == "MemAvailable:") {
            usage.available_memory_bytes = value * 1024;
        }
    }

    usage.used_memory_bytes = usage.total_memory_bytes - usage.available_memory_bytes;
    usage.memory_usage_percent = (usage.used_memory_bytes * 100.0) / usage.total_memory_bytes;

    return usage;
}

ResourceUsage ResourceMonitor::get_disk_info() {
    ResourceUsage usage;
    struct statvfs stat;

    if (statvfs("/", &stat) == 0) {
        usage.total_disk_bytes = stat.f_blocks * stat.f_frsize;
        usage.available_disk_bytes = stat.f_bavail * stat.f_frsize; // Use f_bavail for available blocks
        usage.used_disk_bytes = usage.total_disk_bytes - usage.available_disk_bytes;
        usage.disk_usage_percent = (usage.used_disk_bytes * 100.0) / usage.total_disk_bytes;
    }

    return usage;
}

ResourceUsage ResourceMonitor::get_network_info() {
    ResourceUsage usage;
#ifdef _WIN32
    // Windows network monitoring not implemented yet
    return usage;
#else
    struct ifaddrs *ifaddr, *ifa;

    if (getifaddrs(&ifaddr) == -1) {
        return usage;
    }

    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr) continue;

        // Skip loopback interface
        if (strcmp(ifa->ifa_name, "lo") == 0) continue;

        // Read network statistics
        std::string rx_path = "/sys/class/net/" + std::string(ifa->ifa_name) + "/statistics/rx_bytes";
        std::string tx_path = "/sys/class/net/" + std::string(ifa->ifa_name) + "/statistics/tx_bytes";

        std::ifstream rx_file(rx_path);
        std::ifstream tx_file(tx_path);

        if (rx_file && tx_file) {
            unsigned long long rx_bytes, tx_bytes;
            rx_file >> rx_bytes;
            tx_file >> tx_bytes;

            usage.network_bytes_received += rx_bytes;
            usage.network_bytes_sent += tx_bytes;
        }
    }

    freeifaddrs(ifaddr);
    return usage;
#endif
}

double ResourceMonitor::get_system_load() {
#ifdef _WIN32
    // Windows load average not implemented
    return 0.0;
#else
    double load[3];
    if (getloadavg(load, 3) != -1) {
        return load[0]; // 1-minute load average
    }
    return 0.0;
#endif
}

void ResourceMonitor::check_thresholds(const ResourceUsage& usage) {
    // CPU checks
    if (usage.cpu_usage_percent >= thresholds_.critical_cpu_threshold) {
        generate_alert(ResourceAlertLevel::CRITICAL, "CPU",
                      "Critical CPU usage detected", usage.cpu_usage_percent, thresholds_.critical_cpu_threshold);
    } else if (usage.cpu_usage_percent >= thresholds_.warning_cpu_threshold) {
        generate_alert(ResourceAlertLevel::WARNING, "CPU",
                      "High CPU usage detected", usage.cpu_usage_percent, thresholds_.warning_cpu_threshold);
    }

    // Memory checks
    if (usage.memory_usage_percent >= thresholds_.critical_memory_threshold) {
        generate_alert(ResourceAlertLevel::CRITICAL, "Memory",
                      "Critical memory usage detected", usage.memory_usage_percent, thresholds_.critical_memory_threshold);
    } else if (usage.memory_usage_percent >= thresholds_.warning_memory_threshold) {
        generate_alert(ResourceAlertLevel::WARNING, "Memory",
                      "High memory usage detected", usage.memory_usage_percent, thresholds_.warning_memory_threshold);
    }

    if (usage.available_memory_bytes <= thresholds_.min_available_memory_bytes) {
        generate_alert(ResourceAlertLevel::EMERGENCY, "Memory",
                      "Available memory critically low", usage.available_memory_bytes / (1024.0 * 1024.0),
                      thresholds_.min_available_memory_bytes / (1024.0 * 1024.0));
    }

    // Disk checks
    if (usage.disk_usage_percent >= thresholds_.max_disk_usage_percent) {
        generate_alert(ResourceAlertLevel::WARNING, "Disk",
                      "High disk usage detected", usage.disk_usage_percent, thresholds_.max_disk_usage_percent);
    }

    if (usage.available_disk_bytes <= thresholds_.min_available_disk_bytes) {
        generate_alert(ResourceAlertLevel::CRITICAL, "Disk",
                      "Available disk space critically low", usage.available_disk_bytes / (1024.0 * 1024.0 * 1024.0),
                      thresholds_.min_available_disk_bytes / (1024.0 * 1024.0 * 1024.0));
    }
}

void ResourceMonitor::generate_alert(ResourceAlertLevel level, const std::string& resource_type,
                                    const std::string& message, double current_value, double threshold_value) {
    ResourceAlert alert{
        level,
        resource_type,
        message,
        current_value,
        threshold_value,
        std::chrono::system_clock::now()
    };

    std::lock_guard<std::mutex> lock(data_mutex_);
    active_alerts_.push_back(alert);
    alert_history_.push_back(alert);
}

void ResourceMonitor::add_usage_sample(const ResourceUsage& usage) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    usage_history_.push_back(usage);

    // Keep only last 1000 samples (adjustable)
    if (usage_history_.size() > 1000) {
        usage_history_.erase(usage_history_.begin());
    }
}

void ResourceMonitor::cleanup_old_data() {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto now = std::chrono::system_clock::now();
    auto one_hour_ago = now - std::chrono::hours(1);

    // Remove old alerts
    alert_history_.erase(
        std::remove_if(alert_history_.begin(), alert_history_.end(),
                      [one_hour_ago](const ResourceAlert& alert) {
                          return alert.timestamp < one_hour_ago;
                      }),
        alert_history_.end()
    );
}

#ifdef _WIN32
double ResourceMonitor::get_cpu_usage_windows() const {
    // Simplified CPU usage for Windows - returns 0 for now
    // Full implementation would use PDH or performance counters
    return 0.0;
}
#endif