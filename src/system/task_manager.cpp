#include "task_manager.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <sstream>

Task::Task(const std::string& id, const std::string& description,
           TaskPriority priority, const TaskRequirements& requirements,
           std::function<TaskResult(const std::vector<TaskChunk>&)> executor)
    : id_(id), description_(description), priority_(priority),
      state_(TaskState::QUEUED), requirements_(requirements),
      executor_(executor), progress_(0.0) {
}

void Task::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == TaskState::QUEUED || state_ == TaskState::PAUSED) {
        state_ = TaskState::RUNNING;
        start_time_ = std::chrono::system_clock::now();
    }
}

void Task::pause() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == TaskState::RUNNING) {
        state_ = TaskState::PAUSED;
    }
}

void Task::resume() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == TaskState::PAUSED) {
        state_ = TaskState::RUNNING;
    }
}

void Task::cancel() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != TaskState::COMPLETED && state_ != TaskState::FAILED) {
        state_ = TaskState::CANCELLED;
        end_time_ = std::chrono::system_clock::now();
    }
}

std::vector<TaskChunk> Task::create_chunks(uint32_t total_size, uint32_t chunk_size) {
    std::vector<TaskChunk> chunks;

    if (!requirements_.can_be_chunked || chunk_size == 0) {
        // Create single chunk
        TaskChunk chunk{0, 0, total_size, std::vector<uint8_t>(total_size), {}};
        chunks.push_back(chunk);
        return chunks;
    }

    uint32_t num_chunks = (total_size + chunk_size - 1) / chunk_size; // Ceiling division
    uint32_t actual_chunk_size = std::min(chunk_size, requirements_.max_chunk_size);

    for (uint32_t i = 0; i < num_chunks; ++i) {
        uint32_t start_idx = i * actual_chunk_size;
        uint32_t end_idx = std::min(start_idx + actual_chunk_size, total_size);
        uint32_t chunk_data_size = end_idx - start_idx;

        TaskChunk chunk{
            i,
            start_idx,
            end_idx,
            std::vector<uint8_t>(chunk_data_size),
            {{"chunk_size", std::to_string(chunk_data_size)}}
        };
        chunks.push_back(chunk);
    }

    return chunks;
}

void Task::set_chunks(const std::vector<TaskChunk>& chunks) {
    std::lock_guard<std::mutex> lock(mutex_);
    chunks_ = chunks;
}

void Task::set_result(const TaskResult& result) {
    std::lock_guard<std::mutex> lock(mutex_);
    result_ = result;
    state_ = result.success ? TaskState::COMPLETED : TaskState::FAILED;
    end_time_ = std::chrono::system_clock::now();
    progress_ = 1.0;
}

TaskManager::TaskManager(std::shared_ptr<ResourceMonitor> resource_monitor)
    : resource_monitor_(resource_monitor), max_concurrent_tasks_(4),
      chunking_enabled_(true), scheduler_running_(false),
      total_tasks_processed_(0), total_tasks_failed_(0) {

    // Set default resource limits
    resource_limits_ = {
        95.0,  // max_cpu_usage_percent (raised for neural network training)
        90.0,  // max_memory_usage_percent
        100 * 1024 * 1024,  // min_available_memory_bytes (100MB)
        90.0,  // max_disk_usage_percent
        1024 * 1024 * 1024   // min_available_disk_bytes (1GB)
    };

    std::cout << "Task Manager initialized with " << max_concurrent_tasks_ << " max concurrent tasks" << std::endl;
}

TaskManager::~TaskManager() {
    stop_scheduler();
}

std::string TaskManager::submit_task(std::shared_ptr<Task> task) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string task_id = task->get_id();
    tasks_[task_id] = task;

    // Add to priority queue
    task_queue_.push({task->get_priority(), task_id});

    task_available_cv_.notify_one();

    std::cout << "Task submitted: " << task_id << " (" << task->get_description() << ")" << std::endl;
    return task_id;
}

std::string TaskManager::submit_task(const std::string& description, TaskPriority priority,
                                    const TaskRequirements& requirements,
                                    std::function<TaskResult(const std::vector<TaskChunk>&)> executor) {
    std::string task_id = generate_task_id();
    auto task = std::make_shared<Task>(task_id, description, priority, requirements, executor);
    return submit_task(task);
}

bool TaskManager::pause_task(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tasks_.find(task_id);
    if (it != tasks_.end()) {
        it->second->pause();
        return true;
    }
    return false;
}

bool TaskManager::resume_task(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tasks_.find(task_id);
    if (it != tasks_.end()) {
        it->second->resume();
        return true;
    }
    return false;
}

bool TaskManager::cancel_task(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tasks_.find(task_id);
    if (it != tasks_.end()) {
        it->second->cancel();
        active_tasks_.erase(task_id);
        return true;
    }
    return false;
}

bool TaskManager::prioritize_task(const std::string& task_id, TaskPriority new_priority) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tasks_.find(task_id);
    if (it != tasks_.end()) {
        // Update priority (simplified - in real implementation would update priority queue)
        // For now, just mark that priority changed - actual queue management would be more complex
        std::cout << "Priority updated for task " << task_id << " to " << static_cast<int>(new_priority) << std::endl;
        return true;
    }
    return false;
}

std::shared_ptr<Task> TaskManager::get_task(const std::string& task_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tasks_.find(task_id);
    return (it != tasks_.end()) ? it->second : nullptr;
}

std::vector<std::string> TaskManager::get_active_task_ids() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::vector<std::string>(active_tasks_.begin(), active_tasks_.end());
}

std::vector<std::string> TaskManager::get_queued_task_ids() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> queued;

    // This is a simplified implementation - in reality we'd need to iterate the priority queue
    for (const auto& pair : tasks_) {
        if (pair.second->get_state() == TaskState::QUEUED) {
            queued.push_back(pair.first);
        }
    }

    return queued;
}

std::unordered_map<TaskState, int> TaskManager::get_task_counts() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::unordered_map<TaskState, int> counts;

    for (const auto& pair : tasks_) {
        counts[pair.second->get_state()]++;
    }

    return counts;
}

void TaskManager::set_resource_limits(const ResourceThresholds& limits) {
    std::lock_guard<std::mutex> lock(mutex_);
    resource_limits_ = limits;
}

bool TaskManager::can_schedule_task(const TaskRequirements& requirements) const {
    if (!resource_monitor_) return true;

    auto current_usage = resource_monitor_->get_current_usage();

    // Check resource availability
    bool memory_ok = (current_usage.available_memory_bytes >= requirements.estimated_memory_mb * 1024 * 1024);
    bool cpu_ok = (current_usage.cpu_usage_percent + requirements.estimated_cpu_percent <= resource_limits_.max_cpu_usage_percent);
    bool disk_ok = (current_usage.available_disk_bytes >= requirements.estimated_disk_mb * 1024 * 1024);

    return memory_ok && cpu_ok && disk_ok;
}

double TaskManager::estimate_task_wait_time(const TaskRequirements& requirements) const {
    if (active_tasks_.size() < static_cast<size_t>(max_concurrent_tasks_)) {
        return 0.0; // Can start immediately
    }

    // Simple estimation based on queued tasks and their priorities
    // In a real implementation, this would be more sophisticated
    return 5.0; // 5 seconds average wait time
}

void TaskManager::start_scheduler() {
    if (scheduler_running_) return;

    scheduler_running_ = true;

    // Start scheduler thread
    std::thread scheduler_thread(&TaskManager::scheduler_loop, this);
    scheduler_thread.detach();

    // Start worker threads
    for (int i = 0; i < max_concurrent_tasks_; ++i) {
        worker_threads_.emplace_back(&TaskManager::worker_thread, this);
    }

    std::cout << "Task scheduler started with " << max_concurrent_tasks_ << " worker threads" << std::endl;
}

void TaskManager::stop_scheduler() {
    if (!scheduler_running_) {
        std::cout << "Task scheduler stop called but not running" << std::endl;
        return;
    }
    
    std::cout << "Task scheduler stopping - active tasks: " << active_tasks_.size() << std::endl;
    scheduler_running_ = false;
    task_available_cv_.notify_all();
    
    // Wait for worker threads to finish
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
    
    std::cout << "Task scheduler stopped" << std::endl;
}

void TaskManager::set_max_concurrent_tasks(int max_tasks) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_concurrent_tasks_ = max_tasks;
}

void TaskManager::set_chunking_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    chunking_enabled_ = enabled;
}

std::unordered_map<std::string, double> TaskManager::get_performance_stats() const {
    std::unordered_map<std::string, double> stats;

    stats["total_tasks_processed"] = static_cast<double>(total_tasks_processed_.load());
    stats["total_tasks_failed"] = static_cast<double>(total_tasks_failed_.load());
    uint64_t processed = total_tasks_processed_.load();
    stats["average_processing_time_ms"] = processed > 0 ?
        total_processing_time_.count() / static_cast<double>(processed) : 0.0;
    stats["active_tasks"] = static_cast<double>(active_tasks_.size());
    stats["queued_tasks"] = static_cast<double>(task_queue_.size());

    return stats;
}

std::vector<std::string> TaskManager::get_resource_warnings() const {
    std::vector<std::string> warnings;

    if (!resource_monitor_) return warnings;

    auto alerts = resource_monitor_->get_active_alerts();
    for (const auto& alert : alerts) {
        std::string warning = "Resource alert: " + alert.resource_type + " - " + alert.message;
        warnings.push_back(warning);
    }

    return warnings;
}

// Private methods

void TaskManager::scheduler_loop() {
    while (scheduler_running_) {
        std::unique_lock<std::mutex> lock(mutex_);

        // Wait for tasks or timeout
        task_available_cv_.wait_for(lock, std::chrono::milliseconds(100), [this]() {
            return !task_queue_.empty() || !scheduler_running_;
        });

        if (!scheduler_running_) break;

        // Check resource availability and schedule tasks
        while (!task_queue_.empty() && active_tasks_.size() < static_cast<size_t>(max_concurrent_tasks_)) {
            auto [priority, task_id] = task_queue_.top();
            task_queue_.pop();

            auto task_it = tasks_.find(task_id);
            if (task_it != tasks_.end() && should_schedule_task(task_it->second)) {
                active_tasks_.insert(task_id);
                task_it->second->start();

                // Notify worker thread (in a real implementation, we'd use a thread pool)
                task_available_cv_.notify_one();
            } else {
                // Put back in queue if can't schedule
                task_queue_.push({priority, task_id});
                break;
            }
        }

        lock.unlock();

        // Handle resource pressure
        handle_resource_pressure();

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void TaskManager::worker_thread() {
    while (scheduler_running_) {
        std::unique_lock<std::mutex> lock(mutex_);

        // Wait for work
        task_available_cv_.wait(lock, [this]() {
            return !active_tasks_.empty() || !scheduler_running_;
        });

        if (!scheduler_running_) break;

        // Find a task to execute
        std::string task_id;
        for (const auto& active_id : active_tasks_) {
            auto task_it = tasks_.find(active_id);
            if (task_it != tasks_.end() && task_it->second->get_state() == TaskState::RUNNING) {
                task_id = active_id;
                break;
            }
        }

        if (!task_id.empty()) {
            auto task = tasks_[task_id];
            lock.unlock();

            // Execute the task
            execute_task(task);

            // Clean up
            lock.lock();
            active_tasks_.erase(task_id);
        }
    }
}

bool TaskManager::should_schedule_task(const std::shared_ptr<Task>& task) const {
    // Check if we can schedule based on resources and current load
    return can_schedule_task(task->get_requirements()) &&
           active_tasks_.size() < static_cast<size_t>(max_concurrent_tasks_);
}

void TaskManager::execute_task(std::shared_ptr<Task> task) {
    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        // Create chunks if needed
        std::vector<TaskChunk> chunks;
        if (chunking_enabled_ && task->can_be_chunked()) {
            // For demonstration, create chunks for a hypothetical 1000-item task
            chunks = task->create_chunks(1000, 100);
        } else {
            // Single chunk
            TaskChunk chunk{0, 0, 1000, std::vector<uint8_t>(1000), {}};
            chunks.push_back(chunk);
        }

        // Execute the task
        TaskResult result = task->get_executor()(chunks);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        result.execution_time = duration;
        task->set_result(result);

        update_statistics(result, duration);

    } catch (const std::exception& e) {
        TaskResult failed_result;
        failed_result.success = false;
        failed_result.error_message = std::string("Exception: ") + e.what();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        failed_result.execution_time = duration;
        task->set_result(failed_result);

        total_tasks_failed_++;
    }
}

std::string TaskManager::generate_task_id() const {
    static std::atomic<uint64_t> counter{0};
    std::stringstream ss;
    ss << "task_" << std::chrono::system_clock::now().time_since_epoch().count()
       << "_" << counter++;
    return ss.str();
}

void TaskManager::update_statistics(const TaskResult& result, std::chrono::milliseconds duration) {
    total_tasks_processed_++;
    if (!result.success) {
        total_tasks_failed_++;
    }
    total_processing_time_ += duration;
}

void TaskManager::handle_resource_pressure() {
    if (!resource_monitor_) return;

    auto alerts = resource_monitor_->get_active_alerts();

    for (const auto& alert : alerts) {
        if (alert.level == ResourceAlertLevel::CRITICAL ||
            alert.level == ResourceAlertLevel::EMERGENCY) {

            // Pause some tasks to relieve pressure
            std::lock_guard<std::mutex> lock(mutex_);

            for (const auto& task_id : active_tasks_) {
                auto task_it = tasks_.find(task_id);
                if (task_it != tasks_.end() &&
                    task_it->second->get_requirements().task_type != TaskType::CRITICAL) {
                    task_it->second->pause();
                    std::cout << "Paused task " << task_id << " due to resource pressure" << std::endl;
                    break; // Pause one task at a time
                }
            }
        }
    }
}