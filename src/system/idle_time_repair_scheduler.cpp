#include "idle_time_repair_scheduler.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>
#include <atomic>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
// X11 headers not available in this environment, using simplified idle detection

IdleTimeRepairScheduler::IdleTimeRepairScheduler(std::shared_ptr<ResourceMonitor> resource_monitor)
    : resource_monitor_(resource_monitor), scheduler_running_(false), repairs_paused_(false),
      cpu_idle_threshold_(10.0), user_activity_timeout_seconds_(0.0), // 0 = resource-aware calculation
      total_repairs_attempted_(0), total_repairs_completed_(0), total_repairs_failed_(0) {

    last_user_activity_ = std::chrono::system_clock::now();

    // Set default repair windows (2 AM - 6 AM daily)
    repair_windows_ = {{std::chrono::hours(2), std::chrono::hours(6)}};

    std::cout << "Idle Time Repair Scheduler initialized" << std::endl;
}

IdleTimeRepairScheduler::~IdleTimeRepairScheduler() {
    stop_scheduler();
}

SystemIdleState IdleTimeRepairScheduler::get_current_idle_state() const {
    if (!resource_monitor_) return SystemIdleState::UNKNOWN;

    auto usage = resource_monitor_->get_current_usage();

    // Check CPU usage threshold
    if (usage.cpu_usage_percent > cpu_idle_threshold_) {
        return SystemIdleState::ACTIVE;
    }

    // Check memory availability (ensure enough free memory for learning)
    const size_t min_memory_mb = 256; // Minimum 256MB free for idle operations
    size_t available_memory_mb = usage.available_memory_bytes / (1024 * 1024);
    if (available_memory_mb < min_memory_mb) {
        return SystemIdleState::ACTIVE;
    }

    // Check user activity timeout
    auto now = std::chrono::system_clock::now();
    auto time_since_activity = std::chrono::duration_cast<std::chrono::seconds>(
        now - last_user_activity_).count();

    // Use adaptive timeout based on system resources and user preference
    double idle_timeout_seconds = user_activity_timeout_seconds_;
    if (idle_timeout_seconds == 0.0) {
        // Adaptive timeout: higher load = shorter timeout (more conservative)
        double system_load = usage.cpu_usage_percent + usage.memory_usage_percent;

        if (system_load > 150.0) {
            idle_timeout_seconds = 300.0; // 5 minutes under very high load
        } else if (system_load > 100.0) {
            idle_timeout_seconds = 240.0; // 4 minutes under high load
        } else if (system_load > 50.0) {
            idle_timeout_seconds = 180.0; // 3 minutes under medium load
        } else {
            idle_timeout_seconds = 120.0; // 2 minutes under low load (more aggressive)
        }
    }

    // System is idle if user has been inactive for the required timeout
    if (time_since_activity > idle_timeout_seconds) {
        return SystemIdleState::IDLE;
    }

    return SystemIdleState::ACTIVE;
}

double IdleTimeRepairScheduler::get_current_idle_percentage() const {
    auto usage = resource_monitor_->get_current_usage();
    double cpu_idle = 100.0 - usage.cpu_usage_percent;

#ifdef _WIN32
    // Windows idle time detection
    LASTINPUTINFO lii;
    lii.cbSize = sizeof(LASTINPUTINFO);
    if (GetLastInputInfo(&lii)) {
        DWORD idle_time = GetTickCount() - lii.dwTime;
        double idle_seconds = idle_time / 1000.0;

        // If user has been idle for more than 60 seconds, consider system idle
        if (idle_seconds > 60.0) {
            return 100.0; // Fully idle
        } else {
            // Combine CPU idle with user activity
            return (cpu_idle + (idle_seconds / 60.0 * 100.0)) / 2.0;
        }
    }
#endif

    // Simplified: just return CPU idle percentage
    return cpu_idle;
}

std::vector<IdleTimeWindow> IdleTimeRepairScheduler::get_idle_history(std::chrono::hours window) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto cutoff = std::chrono::system_clock::now() - window;
    std::vector<IdleTimeWindow> recent_windows;

    for (const auto& window_data : idle_history_) {
        if (window_data.start_time > cutoff) {
            recent_windows.push_back(window_data);
        }
    }

    return recent_windows;
}

std::string IdleTimeRepairScheduler::schedule_repair_task(const std::string& description,
                                                        RepairPriority priority,
                                                        std::function<bool()> repair_function,
                                                        std::chrono::seconds estimated_duration,
                                                        bool requires_restart) {
    std::lock_guard<std::mutex> lock(mutex_);

    RepairTask task;
    task.task_id = generate_task_id();
    task.description = description;
    task.priority = priority;
    task.repair_function = repair_function;
    task.estimated_duration = estimated_duration;
    task.requires_restart = requires_restart;
    task.created_time = std::chrono::system_clock::now();
    task.retry_count = 0;
    task.max_retries = 3; // Default retry count

    repair_tasks_[task.task_id] = task;
    repair_queue_.push({priority, task.task_id});

    std::cout << "Scheduled repair task: " << description << " (priority: " << static_cast<int>(priority) << ")" << std::endl;

    return task.task_id;
}

bool IdleTimeRepairScheduler::cancel_repair_task(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = repair_tasks_.find(task_id);
    if (it != repair_tasks_.end()) {
        repair_tasks_.erase(it);
        std::cout << "Cancelled repair task: " << task_id << std::endl;
        return true;
    }

    return false;
}

std::vector<std::string> IdleTimeRepairScheduler::get_pending_repair_tasks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> pending;

    for (const auto& pair : repair_tasks_) {
        // Simplified: consider all tasks as pending
        // In real implementation, would check completion status
        pending.push_back(pair.first);
    }

    return pending;
}

std::unordered_map<std::string, RepairTask> IdleTimeRepairScheduler::get_repair_task_status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return repair_tasks_;
}

void IdleTimeRepairScheduler::set_idle_thresholds(double cpu_idle_threshold,
                                                double user_activity_timeout_seconds) {
    std::lock_guard<std::mutex> lock(mutex_);
    cpu_idle_threshold_ = cpu_idle_threshold;
    user_activity_timeout_seconds_ = user_activity_timeout_seconds;
}

void IdleTimeRepairScheduler::set_repair_windows(const std::vector<std::pair<std::chrono::hours, std::chrono::hours>>& daily_windows) {
    std::lock_guard<std::mutex> lock(mutex_);
    repair_windows_ = daily_windows;
}

void IdleTimeRepairScheduler::start_scheduler() {
    if (scheduler_running_) return;

    scheduler_running_ = true;
    scheduler_thread_ = std::thread(&IdleTimeRepairScheduler::scheduler_loop, this);
    idle_monitor_thread_ = std::thread(&IdleTimeRepairScheduler::idle_monitor_loop, this);

    std::cout << "Idle Time Repair Scheduler started" << std::endl;
}

void IdleTimeRepairScheduler::stop_scheduler() {
    if (!scheduler_running_) return;

    scheduler_running_ = false;
    scheduler_cv_.notify_all();

    if (scheduler_thread_.joinable()) {
        scheduler_thread_.join();
    }
    if (idle_monitor_thread_.joinable()) {
        idle_monitor_thread_.join();
    }

    std::cout << "Idle Time Repair Scheduler stopped" << std::endl;
}

void IdleTimeRepairScheduler::pause_repairs(bool pause) {
    repairs_paused_ = pause;
    std::cout << "Repair scheduling " << (pause ? "paused" : "resumed") << std::endl;
}

bool IdleTimeRepairScheduler::are_repairs_paused() const {
    return repairs_paused_;
}

bool IdleTimeRepairScheduler::execute_emergency_repair(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = repair_tasks_.find(task_id);
    if (it == repair_tasks_.end()) {
        return false;
    }

    // Execute immediately, bypassing idle checks
    execute_repair_task(task_id);
    return true;
}

std::vector<std::string> IdleTimeRepairScheduler::get_critical_repair_tasks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> critical;

    for (const auto& pair : repair_tasks_) {
        if (pair.second.priority == RepairPriority::CRITICAL) {
            critical.push_back(pair.first);
        }
    }

    return critical;
}

std::unordered_map<std::string, double> IdleTimeRepairScheduler::get_repair_statistics() const {
    std::unordered_map<std::string, double> stats;

    stats["total_repairs_attempted"] = static_cast<double>(total_repairs_attempted_.load());
    stats["total_repairs_completed"] = static_cast<double>(total_repairs_completed_.load());
    stats["total_repairs_failed"] = static_cast<double>(total_repairs_failed_.load());
    stats["repair_success_rate"] = total_repairs_attempted_.load() > 0 ?
        (static_cast<double>(total_repairs_completed_.load()) / total_repairs_attempted_.load()) * 100.0 : 0.0;
    stats["average_repair_time_ms"] = total_repairs_completed_.load() > 0 ?
        total_repair_time_.count() / static_cast<double>(total_repairs_completed_.load()) : 0.0;

    return stats;
}

std::vector<std::string> IdleTimeRepairScheduler::get_scheduler_warnings() const {
    std::vector<std::string> warnings;

    if (repairs_paused_) {
        warnings.push_back("Repair scheduling is currently paused");
    }

    auto critical_tasks = get_critical_repair_tasks();
    if (!critical_tasks.empty()) {
        warnings.push_back("Critical repair tasks pending: " + std::to_string(critical_tasks.size()));
    }

    if (!is_within_repair_window()) {
        warnings.push_back("Current time is outside configured repair windows");
    }

    return warnings;
}

// Private methods

void IdleTimeRepairScheduler::scheduler_loop() {
    while (scheduler_running_) {
        std::unique_lock<std::mutex> lock(mutex_);

        // Wait for next check interval or notification
        scheduler_cv_.wait_for(lock, std::chrono::seconds(30));

        if (!scheduler_running_) break;

        // Check if we can execute repairs
        if (repairs_paused_) {
            continue;
        }

        // Process repair queue
        while (!repair_queue_.empty()) {
            auto [priority, task_id] = repair_queue_.top();

            auto task_it = repair_tasks_.find(task_id);
            if (task_it == repair_tasks_.end()) {
                repair_queue_.pop();
                continue;
            }

            const RepairTask& task = task_it->second;

            // Check if we can execute this task
            if (can_execute_repair(task)) {
                repair_queue_.pop();
                lock.unlock();

                // Execute the repair
                execute_repair_task(task_id);

                lock.lock();
            } else {
                // Cannot execute now, check again later
                break;
            }
        }

        lock.unlock();

        // Cleanup completed tasks
        cleanup_completed_tasks();
    }
}

void IdleTimeRepairScheduler::idle_monitor_loop() {
    while (scheduler_running_) {
        // Update idle state
        update_idle_history();

        // User activity is tracked externally - don't update here
        // The last_user_activity_ should only be updated on actual user input

        std::this_thread::sleep_for(std::chrono::seconds(10));
    }
}

bool IdleTimeRepairScheduler::is_system_idle() const {
    return get_current_idle_state() == SystemIdleState::IDLE;
}

bool IdleTimeRepairScheduler::is_within_repair_window() const {
    // Resource-based idle detection - no time windows needed
    // Repairs can run whenever the system is idle based on resource usage
    return true;
}

bool IdleTimeRepairScheduler::can_execute_repair(const RepairTask& task) const {
    // Critical repairs can run anytime
    if (task.priority == RepairPriority::CRITICAL) {
        return true;
    }

    // Check if system is idle and within repair window
    return is_system_idle() && is_within_repair_window() && check_dependencies(task);
}

void IdleTimeRepairScheduler::execute_repair_task(const std::string& task_id) {
    auto start_time = std::chrono::high_resolution_clock::now();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = repair_tasks_.find(task_id);
        if (it == repair_tasks_.end()) {
            return;
        }

        RepairTask& task = it->second;
        total_repairs_attempted_++;

        std::cout << "Executing repair task: " << task.description << std::endl;

        bool success = false;
        try {
            success = task.repair_function();
        } catch (const std::exception& e) {
            std::cerr << "Repair task failed with exception: " << e.what() << std::endl;
            success = false;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        if (success) {
            total_repairs_completed_++;
            total_repair_time_ += duration;
            std::cout << "Repair task completed successfully in " << duration.count() << "ms" << std::endl;
        } else {
            total_repairs_failed_++;
            task.retry_count++;

            if (task.retry_count < task.max_retries) {
                // Re-queue for retry
                repair_queue_.push({task.priority, task_id});
                std::cout << "Repair task failed, retrying (" << task.retry_count << "/" << task.max_retries << ")" << std::endl;
            } else {
                std::cout << "Repair task failed permanently after " << task.max_retries << " attempts" << std::endl;
            }
        }

        log_repair_activity(task, success, duration);
    }
}

bool IdleTimeRepairScheduler::check_dependencies(const RepairTask& task) const {
    // Simplified: assume no dependencies for now
    // In real implementation, would check if dependent repairs completed
    return true;
}

void IdleTimeRepairScheduler::update_idle_history() {
    static auto last_update = std::chrono::system_clock::now();
    static double last_idle_percentage = 0.0;

    auto now = std::chrono::system_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::seconds>(now - last_update);

    if (time_diff.count() >= 60) { // Update every minute
        IdleTimeWindow window;
        window.start_time = last_update;
        window.end_time = now;
        window.idle_percentage = (get_current_idle_percentage() + last_idle_percentage) / 2.0;
        window.duration = time_diff;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            idle_history_.push_back(window);

            // Keep only last 24 hours of history
            auto cutoff = now - std::chrono::hours(24);
            idle_history_.erase(
                std::remove_if(idle_history_.begin(), idle_history_.end(),
                              [cutoff](const IdleTimeWindow& w) { return w.end_time < cutoff; }),
                idle_history_.end()
            );
        }

        last_update = now;
        last_idle_percentage = window.idle_percentage;
    }
}

void IdleTimeRepairScheduler::cleanup_completed_tasks() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Remove tasks that have completed or failed permanently
    for (auto it = repair_tasks_.begin(); it != repair_tasks_.end(); ) {
        const RepairTask& task = it->second;

        // Simplified: remove tasks that have been retried max times and failed
        if (task.retry_count >= task.max_retries) {
            it = repair_tasks_.erase(it);
        } else {
            ++it;
        }
    }
}

std::string IdleTimeRepairScheduler::generate_task_id() const {
    static std::atomic<uint64_t> counter{0};
    std::stringstream ss;
    ss << "repair_" << std::chrono::system_clock::now().time_since_epoch().count()
       << "_" << counter++;
    return ss.str();
}

void IdleTimeRepairScheduler::log_repair_activity(const RepairTask& task, bool success, std::chrono::milliseconds duration) {
    std::cout << "=== Repair Activity Log ===" << std::endl;
    std::cout << "Task: " << task.description << std::endl;
    std::cout << "Priority: " << static_cast<int>(task.priority) << std::endl;
    std::cout << "Duration: " << duration.count() << "ms" << std::endl;
    std::cout << "Success: " << (success ? "Yes" : "No") << std::endl;
    std::cout << "Retries: " << task.retry_count << "/" << task.max_retries << std::endl;
    std::cout << "==========================" << std::endl;
}