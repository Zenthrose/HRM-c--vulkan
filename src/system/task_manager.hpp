#pragma once

#include <string>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <chrono>
#include <memory>
#include "resource_monitor.hpp"

enum class TaskPriority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    CRITICAL = 3
};

enum class TaskState {
    QUEUED,
    RUNNING,
    PAUSED,
    COMPLETED,
    FAILED,
    CANCELLED
};

enum class TaskType {
    COMPUTATION,
    IO_READ,
    IO_WRITE,
    NETWORK_DOWNLOAD,
    NETWORK_UPLOAD,
    MEMORY_INTENSIVE,
    CPU_INTENSIVE,
    MIXED,
    CRITICAL
};

struct TaskRequirements {
    uint64_t estimated_memory_mb;
    double estimated_cpu_percent;
    uint64_t estimated_disk_mb;
    double estimated_duration_seconds;
    TaskType task_type;
    bool can_be_chunked;
    uint32_t max_chunk_size;
};

struct TaskChunk {
    uint32_t chunk_id;
    uint32_t start_index;
    uint32_t end_index;
    std::vector<uint8_t> data;
    std::unordered_map<std::string, std::string> metadata;
};

struct TaskResult {
    bool success;
    std::string error_message;
    std::vector<TaskChunk> chunks;
    std::chrono::milliseconds execution_time;
    uint64_t memory_peak_mb;
    double cpu_peak_percent;
};

class Task {
public:
    Task(const std::string& id, const std::string& description,
         TaskPriority priority, const TaskRequirements& requirements,
         std::function<TaskResult(const std::vector<TaskChunk>&)> executor);

    // Task identification
    std::string get_id() const { return id_; }
    std::string get_description() const { return description_; }
    TaskPriority get_priority() const { return priority_; }
    TaskState get_state() const { return state_; }
    TaskType get_type() const { return requirements_.task_type; }

    // Task lifecycle
    void start();
    void pause();
    void resume();
    void cancel();
    bool is_completed() const { return state_ == TaskState::COMPLETED; }
    bool is_failed() const { return state_ == TaskState::FAILED; }

    // Chunking support
    bool can_be_chunked() const { return requirements_.can_be_chunked; }
    std::vector<TaskChunk> create_chunks(uint32_t total_size, uint32_t chunk_size);
    void set_chunks(const std::vector<TaskChunk>& chunks);

    // Resource requirements
    const TaskRequirements& get_requirements() const { return requirements_; }

    // Executor access (for task manager)
    const std::function<TaskResult(const std::vector<TaskChunk>&)>& get_executor() const { return executor_; }

    // Results
    TaskResult get_result() const { return result_; }
    void set_result(const TaskResult& result);

    // Progress tracking
    double get_progress() const { return progress_; }
    void set_progress(double progress) { progress_ = progress; }

    // Timing
    std::chrono::system_clock::time_point get_start_time() const { return start_time_; }
    std::chrono::system_clock::time_point get_end_time() const { return end_time_; }

private:
    std::string id_;
    std::string description_;
    TaskPriority priority_;
    TaskState state_;
    TaskRequirements requirements_;
    std::function<TaskResult(const std::vector<TaskChunk>&)> executor_;

    std::vector<TaskChunk> chunks_;
    TaskResult result_;
    double progress_;
    std::chrono::system_clock::time_point start_time_;
    std::chrono::system_clock::time_point end_time_;

    mutable std::mutex mutex_;
};

class TaskManager {
public:
    TaskManager(std::shared_ptr<ResourceMonitor> resource_monitor);
    ~TaskManager();

    // Task submission
    std::string submit_task(std::shared_ptr<Task> task);
    std::string submit_task(const std::string& description, TaskPriority priority,
                           const TaskRequirements& requirements,
                           std::function<TaskResult(const std::vector<TaskChunk>&)> executor);

    // Task control
    bool pause_task(const std::string& task_id);
    bool resume_task(const std::string& task_id);
    bool cancel_task(const std::string& task_id);
    bool prioritize_task(const std::string& task_id, TaskPriority new_priority);

    // Task queries
    std::shared_ptr<Task> get_task(const std::string& task_id) const;
    std::vector<std::string> get_active_task_ids() const;
    std::vector<std::string> get_queued_task_ids() const;
    std::unordered_map<TaskState, int> get_task_counts() const;

    // Resource-aware scheduling
    void set_resource_limits(const ResourceThresholds& limits);
    bool can_schedule_task(const TaskRequirements& requirements) const;
    double estimate_task_wait_time(const TaskRequirements& requirements) const;

    // System control
    void start_scheduler();
    void stop_scheduler();
    void set_max_concurrent_tasks(int max_tasks);
    void set_chunking_enabled(bool enabled);

    // Statistics and monitoring
    std::unordered_map<std::string, double> get_performance_stats() const;
    std::vector<std::string> get_resource_warnings() const;

private:
    std::shared_ptr<ResourceMonitor> resource_monitor_;
    std::unordered_map<std::string, std::shared_ptr<Task>> tasks_;
    std::priority_queue<std::pair<TaskPriority, std::string>> task_queue_;
    std::unordered_set<std::string> active_tasks_;
    std::vector<std::thread> worker_threads_;

    // Configuration
    int max_concurrent_tasks_;
    bool chunking_enabled_;
    ResourceThresholds resource_limits_;

    // Synchronization
    mutable std::mutex mutex_;
    std::condition_variable task_available_cv_;
    std::atomic<bool> scheduler_running_;

    // Statistics
    std::atomic<uint64_t> total_tasks_processed_;
    std::atomic<uint64_t> total_tasks_failed_;
    std::chrono::milliseconds total_processing_time_;

    // Private methods
    void scheduler_loop();
    void worker_thread();
    bool should_schedule_task(const std::shared_ptr<Task>& task) const;
    void execute_task(std::shared_ptr<Task> task);
    std::string generate_task_id() const;
    void update_statistics(const TaskResult& result, std::chrono::milliseconds duration);
    void handle_resource_pressure();
};