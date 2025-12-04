#pragma once

#include "self_modifying_hrm.hpp"
#include "resource_monitor.hpp"
#include "task_manager.hpp"

struct ResourceAwareHRMConfig {
    SelfModifyingHRMConfig base_config;
    bool enable_resource_monitoring;
    bool enable_adaptive_task_management;
    bool enable_chunking_for_large_tasks;
    std::chrono::milliseconds resource_check_interval;
    uint64_t max_memory_per_task_mb;
    double max_cpu_per_task_percent;
};

struct ResourceAwareTask {
    std::string task_id;
    std::string description;
    TaskPriority priority;
    TaskRequirements requirements;
    std::function<TaskResult(const std::vector<TaskChunk>&)> executor;
    bool submitted;
    TaskResult result;
};

class ResourceAwareHRM : public SelfModifyingHRM {
public:
    ResourceAwareHRM(const ResourceAwareHRMConfig& config);
    ~ResourceAwareHRM();

    // Enhanced communication with resource awareness
    CommunicationResult communicate(const std::string& input_message);

    // Resource-aware task management
    std::string submit_resource_aware_task(const std::string& description,
                                         TaskPriority priority,
                                         const TaskRequirements& requirements,
                                         std::function<TaskResult(const std::vector<TaskChunk>&)> executor);

    bool pause_task_due_to_resources(const std::string& task_id);
    bool resume_task_when_resources_available(const std::string& task_id);

    // Resource monitoring interface
    ResourceUsage get_current_resource_usage() const;
    std::vector<ResourceAlert> get_resource_alerts() const;
    bool are_resources_available(const TaskRequirements& requirements) const;

    // Adaptive behavior
    void adapt_to_resource_constraints();
    void optimize_for_current_resources();
    std::vector<std::string> get_resource_optimization_suggestions();

    // System status with resource information
    std::unordered_map<std::string, std::string> get_resource_aware_status();

private:
    ResourceAwareHRMConfig config_;
    std::shared_ptr<ResourceMonitor> resource_monitor_;
    std::shared_ptr<TaskManager> task_manager_;

    // Resource-aware state
    std::vector<ResourceAwareTask> pending_tasks_;
    std::unordered_map<std::string, ResourceAwareTask> active_tasks_;
    bool resource_pressure_mode_;

    // Resource-aware methods
    void initialize_resource_monitoring();
    void handle_resource_alerts();
    void adapt_task_execution_to_resources();
    void implement_resource_aware_chunking(const std::string& task_id);

    // Task lifecycle with resource awareness
    bool can_execute_task_now(const ResourceAwareTask& task) const;
    void submit_pending_tasks();
    void manage_task_priorities_based_on_resources();

    // Resource optimization
    void reduce_memory_usage();
    void optimize_cpu_usage();
    void manage_disk_usage();
    void handle_network_constraints();

    // Emergency resource management
    void enter_resource_conservation_mode();
    void exit_resource_conservation_mode();
    void emergency_task_cancellation();
};