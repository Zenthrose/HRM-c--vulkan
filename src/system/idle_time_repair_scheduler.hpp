#pragma once

#include <string>
#include <vector>
#include <queue>
#include <unordered_map>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <chrono>
#include "resource_monitor.hpp"

enum class SystemIdleState {
    ACTIVE,      // System is actively being used
    IDLE,        // System idle, safe for maintenance
    SLEEPING,    // System in sleep/low-power mode
    UNKNOWN      // Cannot determine system state
};

enum class RepairPriority {
    CRITICAL,    // Must fix immediately, even under load
    HIGH,        // Fix when possible, may interrupt some tasks
    NORMAL,      // Fix during idle time
    LOW,         // Fix only when system is very idle
    DEFERRED     // Can be postponed indefinitely
};

struct RepairTask {
    std::string task_id;
    std::string description;
    RepairPriority priority;
    std::function<bool()> repair_function;  // Returns true if repair successful
    std::chrono::seconds estimated_duration;
    std::vector<std::string> dependencies;  // Other repairs that must complete first
    bool requires_restart;  // If true, system restart needed after repair
    std::chrono::system_clock::time_point created_time;
    int retry_count;
    int max_retries;
};

struct IdleTimeWindow {
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
    double idle_percentage;  // How idle the system was during this window
    std::chrono::seconds duration;
};

class IdleTimeRepairScheduler {
public:
    IdleTimeRepairScheduler(std::shared_ptr<ResourceMonitor> resource_monitor);
    ~IdleTimeRepairScheduler();

    // Idle detection and monitoring
    SystemIdleState get_current_idle_state() const;
    double get_current_idle_percentage() const;
    std::vector<IdleTimeWindow> get_idle_history(std::chrono::hours window) const;

    // Repair task management
    std::string schedule_repair_task(const std::string& description,
                                   RepairPriority priority,
                                   std::function<bool()> repair_function,
                                   std::chrono::seconds estimated_duration = std::chrono::seconds(30),
                                   bool requires_restart = false);

    bool cancel_repair_task(const std::string& task_id);
    std::vector<std::string> get_pending_repair_tasks() const;
    std::unordered_map<std::string, RepairTask> get_repair_task_status() const;

    // Idle time configuration
    void set_idle_thresholds(double cpu_idle_threshold = 10.0,  // CPU usage < 10% = idle
                           double user_activity_timeout_seconds = 300.0); // 5 minutes of no activity
    void set_repair_windows(const std::vector<std::pair<std::chrono::hours, std::chrono::hours>>& daily_windows);

    // Scheduler control
    void start_scheduler();
    void stop_scheduler();
    void pause_repairs(bool pause);
    bool are_repairs_paused() const;

    // Emergency repairs (bypass idle requirements)
    bool execute_emergency_repair(const std::string& task_id);
    std::vector<std::string> get_critical_repair_tasks() const;

    // Statistics and monitoring
    std::unordered_map<std::string, double> get_repair_statistics() const;
    std::vector<std::string> get_scheduler_warnings() const;

private:
    std::shared_ptr<ResourceMonitor> resource_monitor_;
    std::unordered_map<std::string, RepairTask> repair_tasks_;
    std::priority_queue<std::pair<RepairPriority, std::string>> repair_queue_;

    // Idle detection
    double cpu_idle_threshold_;
    double user_activity_timeout_seconds_;
    std::chrono::system_clock::time_point last_user_activity_;
    std::vector<IdleTimeWindow> idle_history_;

    // Scheduling
    std::atomic<bool> scheduler_running_;
    std::atomic<bool> repairs_paused_;
    std::thread scheduler_thread_;
    std::thread idle_monitor_thread_;
    mutable std::mutex mutex_;
    std::condition_variable scheduler_cv_;

    // Repair windows (daily schedule)
    std::vector<std::pair<std::chrono::hours, std::chrono::hours>> repair_windows_;

    // Statistics
    std::atomic<uint64_t> total_repairs_attempted_;
    std::atomic<uint64_t> total_repairs_completed_;
    std::atomic<uint64_t> total_repairs_failed_;
    std::chrono::milliseconds total_repair_time_;

    // Private methods
    void scheduler_loop();
    void idle_monitor_loop();
    bool is_system_idle() const;
    bool is_within_repair_window() const;
    bool can_execute_repair(const RepairTask& task) const;
    void execute_repair_task(const std::string& task_id);
    bool check_dependencies(const RepairTask& task) const;
    void update_idle_history();
    void cleanup_completed_tasks();
    std::string generate_task_id() const;
    void log_repair_activity(const RepairTask& task, bool success, std::chrono::milliseconds duration);
};