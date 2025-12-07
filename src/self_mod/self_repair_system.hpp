#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <chrono>
#include <functional>
#include <filesystem>

namespace fs = std::filesystem;

enum class RepairActionType {
    FILE_MODIFICATION,
    CODE_CHANGE,
    CONFIG_UPDATE,
    DRIVE_OPERATION,
    SYSTEM_CHANGE
};

struct RepairAction {
    RepairActionType type;
    std::string description;
    std::string target_path;
    std::string backup_path;
    std::chrono::system_clock::time_point timestamp;
    bool completed;
    double confidence;
    std::unordered_map<std::string, std::string> parameters;
};

struct RollbackPoint {
    std::string id;
    std::chrono::system_clock::time_point timestamp;
    std::vector<RepairAction> actions;
    std::string system_state_snapshot;
    bool is_stable;
};

class SelfRepairSystem {
private:
    std::vector<RollbackPoint> rollback_history_;
    std::vector<RepairAction> pending_actions_;
    std::string backup_directory_;
    std::string project_root_;
    size_t max_rollback_points_;
    bool auto_rollback_enabled_;
    
    std::string generate_rollback_id() const;
    std::string create_backup(const std::string& file_path) const;
    bool restore_backup(const std::string& backup_path, const std::string& target_path) const;
    std::string capture_system_state() const;
    bool validate_system_integrity() const;
    bool can_safely_modify(const std::string& path) const;
    void cleanup_old_backups();

public:
    SelfRepairSystem(const std::string& project_root, const std::string& backup_dir = "./backups");
    ~SelfRepairSystem();
    
    // Core repair operations
    bool begin_repair_sequence(const std::string& description);
    bool add_repair_action(RepairAction& action);
    bool commit_repair_sequence();
    bool rollback_to_point(const std::string& rollback_id);
    bool rollback_last_operation();
    
    // Drive modification safety
    bool safe_drive_operation(const std::string& operation, const std::vector<std::string>& paths);
    bool validate_drive_modification(const std::string& path) const;
    bool create_drive_backup(const std::string& drive_path) const;
    
    // Monitoring and validation
    bool monitor_system_health();
    std::vector<std::string> get_pending_repairs() const;
    std::vector<RollbackPoint> get_available_rollback_points() const;
    
    // Configuration
    void set_auto_rollback(bool enabled);
    void set_max_rollback_points(size_t max_points);
    void set_backup_directory(const std::string& backup_dir);
    
    // Status and reporting
    bool has_pending_operations() const;
    std::string get_last_rollback_point() const;
    void print_repair_status() const;
    bool emergency_rollback();
    
    // Advanced features
    bool schedule_repair_for_idle_time(const RepairAction& action);
    bool validate_repair_safety(const RepairAction& action) const;
    std::vector<std::string> analyze_repair_history() const;
    bool create_safety_checkpoint(const std::string& description);

private:
    bool execute_repair_action(const RepairAction& action);
    bool execute_file_modification(const RepairAction& action);
    bool execute_code_change(const RepairAction& action);
    bool execute_config_update(const RepairAction& action);
    bool execute_drive_operation(const RepairAction& action);
    void load_rollback_history();
    void save_rollback_history() const;
};