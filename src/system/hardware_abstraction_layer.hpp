#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <chrono>
#include <condition_variable>
#include "resource_monitor.hpp"

enum class ComputeBackend {
    GPU_VULKAN,
    CPU_PARALLEL,
    CPU_SEQUENTIAL,
    AUTO
};

enum class HardwareCapability {
    GPU_COMPUTE,
    CPU_MULTITHREADING,
    LARGE_MEMORY,
    FAST_STORAGE,
    NETWORK_ACCESS
};

struct HardwareProfile {
    std::string gpu_name;
    uint64_t gpu_memory_mb;
    int gpu_compute_units;
    std::string cpu_name;
    int cpu_cores;
    uint64_t system_memory_mb;
    uint64_t available_storage_mb;
    std::vector<HardwareCapability> capabilities;
    bool vulkan_supported;
    bool cuda_supported;
};

struct ComputationTask {
    std::string task_id;
    std::string description;
    ComputeBackend preferred_backend;
    std::function<void()> gpu_task;
    std::function<void()> cpu_task;
    uint64_t estimated_gpu_memory_mb;
    uint64_t estimated_cpu_memory_mb;
    double estimated_duration_seconds;
    int priority; // 0-10, higher = more important
};

enum class OffloadReason {
    GPU_UNAVAILABLE,
    GPU_MEMORY_LOW,
    GPU_OVERLOADED,
    CPU_IDLE,
    POWER_SAVING,
    PERFORMANCE_OPTIMIZATION
};

struct OffloadDecision {
    ComputeBackend selected_backend;
    OffloadReason reason;
    double confidence_score; // 0.0 to 1.0
    std::string explanation;
};

class HardwareAbstractionLayer {
public:
    HardwareAbstractionLayer();
    ~HardwareAbstractionLayer();

    // Hardware detection and profiling
    HardwareProfile detect_hardware();
    bool is_gpu_available();
    bool is_vulkan_supported();
    void update_hardware_profile();

    // Dynamic backend selection
    OffloadDecision select_optimal_backend(const ComputationTask& task);
    ComputeBackend get_current_backend();
    void force_backend(ComputeBackend backend);

    // Task execution with automatic offloading
    void execute_task(const ComputationTask& task);
    void execute_task_async(const ComputationTask& task,
                           std::function<void(const ComputationTask&)> callback);

    // Resource-aware task scheduling
    void queue_task(const ComputationTask& task);
    void process_task_queue();
    size_t get_queued_task_count();

    // Performance monitoring
    std::unordered_map<std::string, double> get_backend_performance_stats();
    double get_gpu_utilization();
    double get_cpu_utilization();

    // Backend switching
    bool switch_to_gpu();
    bool switch_to_cpu();
    bool switch_to_auto();

    // Health monitoring
    bool is_backend_healthy(ComputeBackend backend);
    std::vector<std::string> get_backend_warnings();
    void recover_from_backend_failure(ComputeBackend backend);

private:
    HardwareProfile hardware_profile_;
    ComputeBackend current_backend_;
    std::shared_ptr<ResourceMonitor> resource_monitor_;
    std::vector<ComputationTask> task_queue_;
    std::unordered_map<ComputeBackend, std::atomic<int>> active_tasks_per_backend_;

    // Synchronization
    mutable std::mutex mutex_;
    std::condition_variable task_available_cv_;
    std::atomic<bool> running_;

    // Background processing
    std::thread task_processor_thread_;
    std::thread health_monitor_thread_;

    // Hardware detection
    HardwareProfile detect_gpu_capabilities();
    HardwareProfile detect_cpu_capabilities();
    HardwareProfile detect_system_capabilities();
    bool test_vulkan_support();
    bool test_cuda_support();

    // Backend management
    bool initialize_vulkan_backend();
    bool initialize_cpu_backend();
    void cleanup_backends();

    // Task processing
    void task_processor_loop();
    void health_monitor_loop();
    void execute_on_gpu(const ComputationTask& task);
    void execute_on_cpu(const ComputationTask& task);

    // Decision making
    OffloadDecision analyze_task_requirements(const ComputationTask& task);
    OffloadDecision make_offload_decision(const ComputationTask& task);
    double calculate_backend_score(ComputeBackend backend, const ComputationTask& task);

    // Resource monitoring integration
    bool check_resource_availability(ComputeBackend backend, const ComputationTask& task);
    void update_resource_usage_stats();

    // Error handling and recovery
    void handle_gpu_failure();
    void handle_cpu_failure();
    void log_backend_switch(ComputeBackend from, ComputeBackend to, OffloadReason reason);
};