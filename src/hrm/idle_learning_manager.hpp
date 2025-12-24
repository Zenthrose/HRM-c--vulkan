#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <chrono>
#include <queue>
#include "../system/idle_time_repair_scheduler.hpp"
#include "../system/resource_monitor.hpp"

struct LearningDataPoint {
    std::string input_text;
    std::string response_text;
    float confidence_score;
    std::chrono::system_clock::time_point timestamp;
    std::unordered_map<std::string, std::string> metadata;
};

struct IdleLearningState {
    uint64_t total_learning_sessions;
    uint64_t total_data_points_processed;
    std::chrono::system_clock::time_point last_learning_session;
    float average_learning_duration_seconds;
    std::unordered_map<std::string, std::string> learning_metrics;
    bool learning_active;
    // Coordination with communication learning
    uint64_t insights_shared_from_communication;
    std::chrono::system_clock::time_point last_coordination;
};

class IdleLearningManager {
public:
    IdleLearningManager(std::shared_ptr<IdleTimeRepairScheduler> idle_scheduler,
                       std::shared_ptr<ResourceMonitor> resource_monitor);
    ~IdleLearningManager();

    // Learning data accumulation (during active use)
    void accumulate_learning_data(const LearningDataPoint& data_point);
    void accumulate_conversation_data(const std::string& input, const std::string& response,
                                    float confidence = 0.0f);

    // Idle learning control
    void enable_idle_learning(bool enable);
    bool is_idle_learning_enabled() const;
    void set_learning_parameters(size_t max_data_points = 10000,
                               std::chrono::hours learning_interval = std::chrono::hours(4));

    // Learning execution (called during idle periods)
    bool perform_idle_learning_session();
    bool process_accumulated_data();
    bool update_learning_models();

    // State persistence
    bool save_learning_state(const std::string& state_file = "idle_learning_state.json");
    bool load_learning_state(const std::string& state_file = "idle_learning_state.json");

    // Monitoring and statistics
    IdleLearningState get_learning_state() const;
    std::vector<std::string> get_learning_status() const;
    size_t get_pending_data_points() const;
    std::unordered_map<std::string, double> get_learning_metrics() const;

    // Resource management
    void set_resource_limits(double max_cpu_percent = 20.0, size_t max_memory_mb = 512);
    bool check_resource_availability() const;

    // Coordination with communication learning
    void coordinate_with_communication_learning();
    bool should_coordinate_now() const;
    uint64_t get_shared_insights_count() const;

private:
    std::shared_ptr<IdleTimeRepairScheduler> idle_scheduler_;
    std::shared_ptr<ResourceMonitor> resource_monitor_;

    // Learning data storage
    std::queue<LearningDataPoint> accumulated_data_;
    mutable std::mutex data_mutex_;
    size_t max_data_points_;
    std::chrono::hours learning_interval_;

    // Learning state
    IdleLearningState learning_state_;
    mutable std::mutex state_mutex_;
    std::atomic<bool> learning_enabled_;
    std::string state_file_path_;

    // Resource limits
    double max_cpu_percent_;
    size_t max_memory_mb_;

    // Learning session management
    std::atomic<bool> session_in_progress_;
    std::chrono::system_clock::time_point last_session_time_;

    // Private methods
    void schedule_learning_task();
    bool validate_learning_conditions() const;
    void update_learning_metrics(const std::chrono::milliseconds& duration, bool success);
    void cleanup_old_data();
    bool compress_data_if_needed();
    std::vector<LearningDataPoint> get_batch_for_processing(size_t batch_size = 100);
    void log_learning_activity(const std::string& activity, bool success = true);
};