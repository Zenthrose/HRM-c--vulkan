#include "resource_aware_hrm.hpp"
#include <iostream>
#include <algorithm>

ResourceAwareHRM::ResourceAwareHRM(const ResourceAwareHRMConfig& config)
    : SelfModifyingHRM(config.base_config), config_(config), resource_pressure_mode_(false) {
    std::cout << "Initializing Resource-Aware HRM System..." << std::endl;

    // Initialize resource monitoring
    resource_monitor_ = std::make_shared<ResourceMonitor>();

    // Initialize task manager
    task_manager_ = std::make_shared<TaskManager>(resource_monitor_);

    // Set up resource monitoring
    if (config.enable_resource_monitoring) {
        initialize_resource_monitoring();
    }

    // Start task manager
    if (config.enable_adaptive_task_management) {
        task_manager_->start_scheduler();
    }

    std::cout << "Resource-Aware HRM System initialized with "
              << (config.enable_resource_monitoring ? "resource monitoring" : "no resource monitoring")
              << " and "
              << (config.enable_adaptive_task_management ? "adaptive task management" : "basic task management")
              << std::endl;
}

ResourceAwareHRM::~ResourceAwareHRM() {
    if (task_manager_) {
        task_manager_->stop_scheduler();
    }
    if (resource_monitor_) {
        resource_monitor_->stop_monitoring();
    }
}

CommunicationResult ResourceAwareHRM::communicate(const std::string& input_message) {
    // Check resource availability before processing
    if (config_.enable_resource_monitoring) {
        auto current_usage = resource_monitor_->get_current_usage();

        // Check for resource pressure
        if (current_usage.memory_usage_percent > 85.0 ||
            current_usage.cpu_usage_percent > 80.0) {
            resource_pressure_mode_ = true;
            adapt_to_resource_constraints();
        } else {
            resource_pressure_mode_ = false;
        }

        // Handle resource alerts
        handle_resource_alerts();
    }

    // Perform normal communication
    CommunicationResult result = SelfModifyingHRM::communicate(input_message);

    // Submit any pending resource-aware tasks
    if (config_.enable_adaptive_task_management) {
        submit_pending_tasks();
        adapt_task_execution_to_resources();
    }

    return result;
}

std::string ResourceAwareHRM::submit_resource_aware_task(const std::string& description,
                                                       TaskPriority priority,
                                                       const TaskRequirements& requirements,
                                                       std::function<TaskResult(const std::vector<TaskChunk>&)> executor) {
    ResourceAwareTask task;
    task.task_id = "res_aware_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    task.description = description;
    task.priority = priority;
    task.requirements = requirements;
    task.executor = executor;
    task.submitted = false;

    // Check if we can execute immediately
    if (can_execute_task_now(task)) {
        std::string task_id = task_manager_->submit_task(task.description, task.priority,
                                                       task.requirements, task.executor);
        task.submitted = true;
        active_tasks_[task_id] = task;
        return task_id;
    } else {
        // Queue for later execution
        pending_tasks_.push_back(task);
        std::cout << "Task queued due to resource constraints: " << description << std::endl;
        return task.task_id;
    }
}

bool ResourceAwareHRM::pause_task_due_to_resources(const std::string& task_id) {
    if (task_manager_->pause_task(task_id)) {
        std::cout << "Paused task due to resource constraints: " << task_id << std::endl;
        return true;
    }
    return false;
}

bool ResourceAwareHRM::resume_task_when_resources_available(const std::string& task_id) {
    if (are_resources_available(active_tasks_[task_id].requirements)) {
        if (task_manager_->resume_task(task_id)) {
            std::cout << "Resumed task - resources now available: " << task_id << std::endl;
            return true;
        }
    }
    return false;
}

ResourceUsage ResourceAwareHRM::get_current_resource_usage() const {
    return resource_monitor_->get_current_usage();
}

std::vector<ResourceAlert> ResourceAwareHRM::get_resource_alerts() const {
    return resource_monitor_->get_active_alerts();
}

bool ResourceAwareHRM::are_resources_available(const TaskRequirements& requirements) const {
    return task_manager_->can_schedule_task(requirements);
}

void ResourceAwareHRM::adapt_to_resource_constraints() {
    std::cout << "Adapting to resource constraints..." << std::endl;

    // Reduce memory usage
    reduce_memory_usage();

    // Optimize CPU usage
    optimize_cpu_usage();

    // Manage task priorities
    manage_task_priorities_based_on_resources();

    // Enter conservation mode if needed
    auto alerts = get_resource_alerts();
    bool has_critical_alerts = std::any_of(alerts.begin(), alerts.end(),
        [](const ResourceAlert& alert) {
            return alert.level == ResourceAlertLevel::CRITICAL ||
                   alert.level == ResourceAlertLevel::EMERGENCY;
        });

    if (has_critical_alerts) {
        enter_resource_conservation_mode();
    }
}

void ResourceAwareHRM::optimize_for_current_resources() {
    auto current_usage = get_current_resource_usage();

    // Adjust chunking based on available memory
    if (current_usage.available_memory_bytes < 500 * 1024 * 1024) { // Less than 500MB
        task_manager_->set_chunking_enabled(true);
        std::cout << "Enabled aggressive chunking due to low memory" << std::endl;
    } else {
        task_manager_->set_chunking_enabled(config_.enable_chunking_for_large_tasks);
    }

    // Adjust concurrent tasks based on CPU
    if (current_usage.cpu_usage_percent > 70.0) {
        task_manager_->set_max_concurrent_tasks(2);
    } else if (current_usage.cpu_usage_percent < 30.0) {
        task_manager_->set_max_concurrent_tasks(6);
    } else {
        task_manager_->set_max_concurrent_tasks(4);
    }
}

std::vector<std::string> ResourceAwareHRM::get_resource_optimization_suggestions() {
    std::vector<std::string> suggestions;
    auto current_usage = get_current_resource_usage();

    if (current_usage.memory_usage_percent > 80.0) {
        suggestions.push_back("High memory usage detected - consider enabling chunking for large tasks");
        suggestions.push_back("Reduce concurrent task count to free memory");
    }

    if (current_usage.cpu_usage_percent > 75.0) {
        suggestions.push_back("High CPU usage - consider pausing non-critical tasks");
        suggestions.push_back("Enable CPU-intensive task throttling");
    }

    if (current_usage.disk_usage_percent > 85.0) {
        suggestions.push_back("Low disk space - clean up temporary files and cache");
        suggestions.push_back("Disable disk-intensive operations");
    }

    if (pending_tasks_.size() > 5) {
        suggestions.push_back("Many tasks queued - consider increasing resource limits or optimizing task requirements");
    }

    return suggestions;
}

std::unordered_map<std::string, std::string> ResourceAwareHRM::get_resource_aware_status() {
    auto base_status = SelfModifyingHRM::get_self_analysis_report();

    // Add resource-specific information
    auto resource_usage = get_current_resource_usage();
    base_status["memory_usage_percent"] = std::to_string(resource_usage.memory_usage_percent);
    base_status["cpu_usage_percent"] = std::to_string(resource_usage.cpu_usage_percent);
    base_status["disk_usage_percent"] = std::to_string(resource_usage.disk_usage_percent);
    base_status["available_memory_mb"] = std::to_string(resource_usage.available_memory_bytes / (1024 * 1024));
    base_status["resource_pressure_mode"] = resource_pressure_mode_ ? "true" : "false";
    base_status["pending_tasks"] = std::to_string(pending_tasks_.size());
    base_status["active_tasks"] = std::to_string(active_tasks_.size());

    // Add task manager stats
    auto task_stats = task_manager_->get_performance_stats();
    for (const auto& stat : task_stats) {
        base_status["task_" + stat.first] = std::to_string(stat.second);
    }

    return base_status;
}

// Private methods

void ResourceAwareHRM::initialize_resource_monitoring() {
    resource_monitor_->start_monitoring(config_.resource_check_interval);

    // Set resource limits for task manager
    ResourceThresholds limits;
    limits.max_memory_usage_percent = 85.0;
    limits.max_cpu_usage_percent = 80.0;
    limits.min_available_memory_bytes = config_.max_memory_per_task_mb * 1024 * 1024;
    limits.max_disk_usage_percent = 90.0;
    limits.min_available_disk_bytes = 1024 * 1024 * 1024; // 1GB

    task_manager_->set_resource_limits(limits);
}

void ResourceAwareHRM::handle_resource_alerts() {
    auto alerts = get_resource_alerts();

    for (const auto& alert : alerts) {
        switch (alert.level) {
            case ResourceAlertLevel::WARNING:
                std::cout << "Resource warning: " << alert.message << std::endl;
                // Pause some tasks
                manage_task_priorities_based_on_resources();
                break;

            case ResourceAlertLevel::CRITICAL:
                std::cout << "Resource critical: " << alert.message << std::endl;
                // More aggressive task management
                enter_resource_conservation_mode();
                break;

            case ResourceAlertLevel::EMERGENCY:
                std::cout << "Resource emergency: " << alert.message << std::endl;
                // Emergency measures
                emergency_task_cancellation();
                break;

            default:
                break;
        }
    }
}

void ResourceAwareHRM::adapt_task_execution_to_resources() {
    // Resume tasks that can now run
    for (auto it = pending_tasks_.begin(); it != pending_tasks_.end(); ) {
        if (can_execute_task_now(*it)) {
            std::string task_id = task_manager_->submit_task(it->description, it->priority,
                                                           it->requirements, it->executor);
            it->submitted = true;
            active_tasks_[task_id] = *it;
            it = pending_tasks_.erase(it);
            std::cout << "Submitted pending task: " << it->description << std::endl;
        } else {
            ++it;
        }
    }
}

void ResourceAwareHRM::implement_resource_aware_chunking(const std::string& task_id) {
    // This would implement intelligent chunking based on available resources
    // For now, it's a placeholder
    std::cout << "Resource-aware chunking for task: " << task_id << std::endl;
}

bool ResourceAwareHRM::can_execute_task_now(const ResourceAwareTask& task) const {
    return are_resources_available(task.requirements);
}

void ResourceAwareHRM::submit_pending_tasks() {
    // Submit tasks that can now run
    adapt_task_execution_to_resources();
}

void ResourceAwareHRM::manage_task_priorities_based_on_resources() {
    auto current_usage = get_current_resource_usage();

    // Increase priority of memory-light tasks when memory is low
    if (current_usage.memory_usage_percent > 70.0) {
        // This would require more sophisticated task priority management
        std::cout << "Adjusting task priorities due to high memory usage" << std::endl;
    }
}

void ResourceAwareHRM::reduce_memory_usage() {
    // Force garbage collection or memory cleanup
    std::cout << "Reducing memory usage..." << std::endl;

    // Clear any cached data
    // Reduce buffer sizes
    // This would be system-specific
}

void ResourceAwareHRM::optimize_cpu_usage() {
    // Throttle CPU-intensive operations
    std::cout << "Optimizing CPU usage..." << std::endl;

    // Reduce thread counts
    // Add delays between operations
    // This would be system-specific
}

void ResourceAwareHRM::manage_disk_usage() {
    // Clean up temporary files
    std::cout << "Managing disk usage..." << std::endl;

    // Remove old temp files
    // Compress data
    // This would be system-specific
}

void ResourceAwareHRM::handle_network_constraints() {
    // Throttle network operations
    std::cout << "Handling network constraints..." << std::endl;

    // Reduce download speeds
    // Pause network tasks
    // This would be system-specific
}

void ResourceAwareHRM::enter_resource_conservation_mode() {
    std::cout << "Entering resource conservation mode" << std::endl;

    // Aggressive resource management
    task_manager_->set_max_concurrent_tasks(1);
    task_manager_->set_chunking_enabled(true);

    // Pause non-critical tasks
    auto active_ids = task_manager_->get_active_task_ids();
    for (const auto& task_id : active_ids) {
        auto task = task_manager_->get_task(task_id);
        if (task && task->get_requirements().task_type != TaskType::CRITICAL) {
            task_manager_->pause_task(task_id);
        }
    }
}

void ResourceAwareHRM::exit_resource_conservation_mode() {
    std::cout << "Exiting resource conservation mode" << std::endl;

    // Restore normal operation
    task_manager_->set_max_concurrent_tasks(4);
    task_manager_->set_chunking_enabled(config_.enable_chunking_for_large_tasks);

    // Resume paused tasks
    auto active_ids = task_manager_->get_active_task_ids();
    for (const auto& task_id : active_ids) {
        task_manager_->resume_task(task_id);
    }
}

void ResourceAwareHRM::emergency_task_cancellation() {
    std::cout << "Emergency task cancellation initiated" << std::endl;

    // Cancel non-critical tasks
    auto active_ids = task_manager_->get_active_task_ids();
    for (const auto& task_id : active_ids) {
        auto task = task_manager_->get_task(task_id);
        if (task && task->get_requirements().task_type != TaskType::CRITICAL) {
            task_manager_->cancel_task(task_id);
        }
    }
}