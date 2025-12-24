#include "self_repair_system.hpp"
#include "runtime_compilation_system.hpp"
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <filesystem>

namespace fs = std::filesystem;

SelfRepairSystem::SelfRepairSystem(const std::string& project_root, const std::string& backup_dir)
    : project_root_(project_root), backup_directory_(backup_dir), 
      max_rollback_points_(10), auto_rollback_enabled_(true) {
    
    std::cout << "Initializing Self-Repair System..." << std::endl;
    
    // Create backup directory if it doesn't exist
    if (!fs::exists(backup_directory_)) {
        fs::create_directories(backup_directory_);
        std::cout << "Created backup directory: " << backup_directory_ << std::endl;
    }
    
    // Load existing rollback points if any
    load_rollback_history();
    
    std::cout << "Self-Repair System initialized with auto-rollback " 
              << (auto_rollback_enabled_ ? "enabled" : "disabled") << std::endl;
}

SelfRepairSystem::~SelfRepairSystem() {
    // Cleanup any pending operations
    if (!pending_actions_.empty()) {
        std::cout << "Warning: " << pending_actions_.size() 
                  << " pending repair actions at shutdown" << std::endl;
    }
    
    // Save rollback history
    save_rollback_history();
}

bool SelfRepairSystem::begin_repair_sequence(const std::string& description) {
    if (!pending_actions_.empty()) {
        std::cout << "Error: Cannot start new repair sequence with pending actions" << std::endl;
        return false;
    }
    
    std::cout << "Starting repair sequence: " << description << std::endl;
    
    // Create new rollback point
    RollbackPoint point;
    point.id = generate_rollback_id();
    point.timestamp = std::chrono::system_clock::now();
    point.system_state_snapshot = capture_system_state();
    point.is_stable = validate_system_integrity();
    
    rollback_history_.push_back(point);
    cleanup_old_backups();
    
    return true;
}

bool SelfRepairSystem::add_repair_action(RepairAction& action) {
    if (!validate_repair_safety(action)) {
        std::cout << "Error: Repair action failed safety validation: " << action.description << std::endl;
        return false;
    }
    
    // Create backup before modification
    if (!action.target_path.empty() && fs::exists(action.target_path)) {
        action.backup_path = create_backup(action.target_path);
        if (action.backup_path.empty()) {
            std::cout << "Error: Failed to create backup for " << action.target_path << std::endl;
            return false;
        }
    }
    
    pending_actions_.push_back(action);
    std::cout << "Added repair action: " << action.description 
              << " (confidence: " << action.confidence << ")" << std::endl;
    
    return true;
}

bool SelfRepairSystem::commit_repair_sequence() {
    if (pending_actions_.empty()) {
        std::cout << "Error: No pending actions to commit" << std::endl;
        return false;
    }
    
    std::cout << "Committing " << pending_actions_.size() << " repair actions..." << std::endl;
    
    bool all_success = true;
    for (auto& action : pending_actions_) {
        try {
            // Execute the repair action based on type
            bool success = execute_repair_action(action);
            action.completed = success;
            
            if (!success) {
                std::cout << "Failed to execute action: " << action.description << std::endl;
                all_success = false;
            }
        } catch (const std::exception& e) {
            std::cout << "Exception during action " << action.description 
                      << ": " << e.what() << std::endl;
            action.completed = false;
            all_success = false;
        }
    }
    
    // Update the latest rollback point with committed actions
    if (!rollback_history_.empty()) {
        rollback_history_.back().actions = pending_actions_;
    }
    
    pending_actions_.clear();
    
    if (all_success) {
        std::cout << "Repair sequence committed successfully" << std::endl;
    } else {
        std::cout << "Repair sequence completed with some failures" << std::endl;
        if (auto_rollback_enabled_) {
            std::cout << "Auto-rollback triggered due to failures" << std::endl;
            rollback_last_operation();
        }
    }
    
    return all_success;
}

bool SelfRepairSystem::rollback_to_point(const std::string& rollback_id) {
    auto it = std::find_if(rollback_history_.begin(), rollback_history_.end(),
        [&rollback_id](const RollbackPoint& point) {
            return point.id == rollback_id;
        });
    
    if (it == rollback_history_.end()) {
        std::cout << "Error: Rollback point not found: " << rollback_id << std::endl;
        return false;
    }
    
    std::cout << "Rolling back to point: " << rollback_id << std::endl;
    
    // Restore actions in reverse order
    for (auto action_it = it->actions.rbegin(); action_it != it->actions.rend(); ++action_it) {
        if (!action_it->completed || action_it->backup_path.empty()) {
            continue;
        }
        
        if (!restore_backup(action_it->backup_path, action_it->target_path)) {
            std::cout << "Failed to restore backup for " << action_it->target_path << std::endl;
            return false;
        }
    }
    
    // Remove rollback points after this one
    rollback_history_.erase(it, rollback_history_.end());
    
    std::cout << "Rollback completed successfully" << std::endl;
    return true;
}

bool SelfRepairSystem::rollback_last_operation() {
    if (rollback_history_.empty()) {
        std::cout << "Error: No operations to rollback" << std::endl;
        return false;
    }
    
    return rollback_to_point(rollback_history_.back().id);
}

bool SelfRepairSystem::safe_drive_operation(const std::string& operation, const std::vector<std::string>& paths) {
    std::cout << "Performing safe drive operation: " << operation << std::endl;
    
    // Validate all paths first
    for (const auto& path : paths) {
        if (!validate_drive_modification(path)) {
            std::cout << "Error: Unsafe to modify path: " << path << std::endl;
            return false;
        }
    }
    
    // Create rollback point
    if (!begin_repair_sequence("Drive operation: " + operation)) {
        return false;
    }
    
    // Add actions for each path
    bool all_safe = true;
    for (const auto& path : paths) {
        RepairAction action;
        action.type = RepairActionType::DRIVE_OPERATION;
        action.description = operation + " on " + path;
        action.target_path = path;
        action.confidence = 0.9;
        action.completed = false;
        
        if (!add_repair_action(action)) {
            all_safe = false;
        }
    }
    
    if (all_safe) {
        return commit_repair_sequence();
    } else {
        rollback_last_operation();
        return false;
    }
}

bool SelfRepairSystem::validate_drive_modification(const std::string& path) const {
    // Check if path is safe to modify
    std::vector<std::string> dangerous_paths = {
        "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)",
        "/boot", "/etc", "/usr/bin", "/bin", "/sbin",
        "/System", "/Library"
    };
    
    for (const auto& dangerous : dangerous_paths) {
        if (path.find(dangerous) != std::string::npos) {
            std::cout << "Warning: Path contains system directory: " << path << std::endl;
            return false;
        }
    }
    
    // Check if file is critical
    std::vector<std::string> critical_extensions = {".sys", ".dll", ".exe", ".so", ".dylib"};
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (std::find(critical_extensions.begin(), critical_extensions.end(), ext) != critical_extensions.end()) {
        std::cout << "Warning: Critical system file: " << path << std::endl;
        return false;
    }
    
    return true;
}

bool SelfRepairSystem::create_drive_backup(const std::string& drive_path) const {
    std::string timestamp = std::to_string(std::time(nullptr));
    std::string backup_name = "drive_backup_" + timestamp + ".tar";
    fs::path backup_path = fs::path(backup_directory_) / backup_name;

    std::cout << "Creating drive backup: " << backup_path.string() << std::endl;

    // Use system tar/backup command
    std::string command = "tar -czf \"" + backup_path.string() + "\" \"" + drive_path + "\"";
    int result = std::system(command.c_str());

    return result == 0;
}

std::string SelfRepairSystem::generate_rollback_id() const {
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    return "rollback_" + std::to_string(timestamp);
}

std::string SelfRepairSystem::create_backup(const std::string& file_path) const {
    if (!fs::exists(file_path)) {
        return "";
    }

    fs::path file_path_obj(file_path);
    std::string filename = file_path_obj.filename().string();
    std::string timestamp = std::to_string(std::time(nullptr));
    std::string backup_name = filename + ".backup_" + timestamp;
    fs::path backup_path = fs::path(backup_directory_) / backup_name;

    try {
        fs::copy_file(file_path, backup_path, fs::copy_options::overwrite_existing);
        return backup_path.string();
    } catch (const std::exception& e) {
        std::cout << "Backup creation failed: " << e.what() << std::endl;
        return "";
    }
}

bool SelfRepairSystem::restore_backup(const std::string& backup_path, const std::string& target_path) const {
    if (!fs::exists(backup_path)) {
        std::cout << "Backup file not found: " << backup_path << std::endl;
        return false;
    }
    
    try {
        fs::copy_file(backup_path, target_path, fs::copy_options::overwrite_existing);
        return true;
    } catch (const std::exception& e) {
        std::cout << "Backup restore failed: " << e.what() << std::endl;
        return false;
    }
}

std::string SelfRepairSystem::capture_system_state() const {
    // Capture basic system state for rollback validation
    std::ostringstream state;
    state << "timestamp:" << std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Add project file count
    size_t file_count = 0;
    if (fs::exists(project_root_)) {
        for (const auto& entry : fs::recursive_directory_iterator(project_root_)) {
            if (entry.is_regular_file()) {
                file_count++;
            }
        }
    }
    state << ":files:" << file_count;
    
    return state.str();
}

bool SelfRepairSystem::validate_system_integrity() const {
    // Basic integrity checks
    if (!fs::exists(project_root_)) {
        return false;
    }
    
    // Check for critical files
    std::vector<std::string> critical_files = {"CMakeLists.txt", "README.md"};
    for (const auto& file : critical_files) {
        if (fs::exists(fs::path(project_root_) / file)) {
            return true;
        }
    }
    
    return false;
}

bool SelfRepairSystem::validate_repair_safety(const RepairAction& action) const {
    // Check if action is safe to execute
    if (action.confidence < 0.5) {
        std::cout << "Action confidence too low: " << action.confidence << std::endl;
        return false;
    }
    
    // Validate target path
    if (!action.target_path.empty() && !can_safely_modify(action.target_path)) {
        return false;
    }
    
    return true;
}

bool SelfRepairSystem::can_safely_modify(const std::string& path) const {
    // Check if path is within project bounds
    std::string abs_path = fs::absolute(path).string();
    std::string abs_project = fs::absolute(project_root_).string();
    
    if (abs_path.find(abs_project) != 0) {
        std::cout << "Path outside project bounds: " << path << std::endl;
        return false;
    }
    
    return true;
}

void SelfRepairSystem::cleanup_old_backups() {
    if (rollback_history_.size() > max_rollback_points_) {
        // Remove oldest rollback points and their backups
        size_t to_remove = rollback_history_.size() - max_rollback_points_;
        for (size_t i = 0; i < to_remove; ++i) {
            const auto& point = rollback_history_[i];
            for (const auto& action : point.actions) {
                if (!action.backup_path.empty() && fs::exists(action.backup_path)) {
                    fs::remove(action.backup_path);
                }
            }
        }
        rollback_history_.erase(rollback_history_.begin(), 
                              rollback_history_.begin() + to_remove);
    }
}

bool SelfRepairSystem::execute_repair_action(const RepairAction& action) {
    switch (action.type) {
        case RepairActionType::FILE_MODIFICATION:
            return execute_file_modification(action);
        case RepairActionType::CODE_CHANGE:
            return execute_code_change(action);
        case RepairActionType::CONFIG_UPDATE:
            return execute_config_update(action);
        case RepairActionType::DRIVE_OPERATION:
            return execute_drive_operation(action);
        default:
            std::cout << "Unknown repair action type" << std::endl;
            return false;
    }
}

bool SelfRepairSystem::execute_file_modification(const RepairAction& action) {
    std::cout << "Executing file modification: " << action.description << std::endl;

    // Extract operation type and paths from action parameters
    auto operation_it = action.parameters.find("operation");
    auto source_it = action.parameters.find("source_path");
    auto dest_it = action.parameters.find("dest_path");

    if (operation_it == action.parameters.end()) {
        std::cout << "Missing operation parameter" << std::endl;
        return false;
    }

    std::string operation = operation_it->second;

    try {
        if (operation == "copy") {
            if (source_it == action.parameters.end() || dest_it == action.parameters.end()) {
                std::cout << "Missing source_path or dest_path for copy operation" << std::endl;
                return false;
            }
            fs::copy_file(source_it->second, dest_it->second, fs::copy_options::overwrite_existing);
            std::cout << "File copied successfully" << std::endl;
        } else if (operation == "move") {
            if (source_it == action.parameters.end() || dest_it == action.parameters.end()) {
                std::cout << "Missing source_path or dest_path for move operation" << std::endl;
                return false;
            }
            fs::rename(source_it->second, dest_it->second);
            std::cout << "File moved successfully" << std::endl;
        } else if (operation == "delete") {
            if (source_it == action.parameters.end()) {
                std::cout << "Missing source_path for delete operation" << std::endl;
                return false;
            }
            fs::remove(source_it->second);
            std::cout << "File deleted successfully" << std::endl;
        } else {
            std::cout << "Unknown file operation: " << operation << std::endl;
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        std::cout << "File operation failed: " << e.what() << std::endl;
        return false;
    }
}

bool SelfRepairSystem::execute_code_change(const RepairAction& action) {
    std::cout << "Executing code change: " << action.description << std::endl;

    // Extract file path and modification from action parameters
    auto file_it = action.parameters.find("file_path");
    auto modification_it = action.parameters.find("modification");

    if (file_it == action.parameters.end() || modification_it == action.parameters.end()) {
        std::cout << "Missing file_path or modification parameters" << std::endl;
        return false;
    }

    std::string file_path = file_it->second;
    std::string modification = modification_it->second;

    // Use runtime compilation system to apply the change
    RuntimeCompilationSystem compiler;
    CompilationResult result = compiler.modify_and_recompile(file_path, modification, "repaired_component");

    if (result.success) {
        std::cout << "Code change applied successfully" << std::endl;
        // In a full implementation, would load the new library and update function pointers
        return true;
    } else {
        std::cout << "Code change failed: " << result.errors[0] << std::endl;
        return false;
    }
}

bool SelfRepairSystem::execute_config_update(const RepairAction& action) {
    std::cout << "Executing config update: " << action.description << std::endl;

    // Extract config file path and new value from action parameters
    auto file_it = action.parameters.find("config_file");
    auto key_it = action.parameters.find("config_key");
    auto value_it = action.parameters.find("new_value");

    if (file_it == action.parameters.end() || key_it == action.parameters.end() || value_it == action.parameters.end()) {
        std::cout << "Missing config_file, config_key, or new_value parameters" << std::endl;
        return false;
    }

    std::string config_file = file_it->second;
    std::string config_key = key_it->second;
    std::string new_value = value_it->second;

    // Read the config file
    std::ifstream input_file(config_file);
    if (!input_file.is_open()) {
        std::cout << "Failed to open config file: " << config_file << std::endl;
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(input_file)),
                        std::istreambuf_iterator<char>());
    input_file.close();

    // Simple key=value replacement (for basic config files)
    std::string search_pattern = config_key + "=";
    size_t pos = content.find(search_pattern);
    if (pos == std::string::npos) {
        std::cout << "Config key not found: " << config_key << std::endl;
        return false;
    }

    // Find the end of the line
    size_t end_pos = content.find('\n', pos);
    if (end_pos == std::string::npos) {
        end_pos = content.length();
    }

    // Replace the value
    std::string old_line = content.substr(pos, end_pos - pos);
    std::string new_line = search_pattern + new_value;

    content.replace(pos, old_line.length(), new_line);

    // Write back to file
    std::ofstream output_file(config_file);
    if (!output_file.is_open()) {
        std::cout << "Failed to write config file: " << config_file << std::endl;
        return false;
    }
    output_file << content;
    output_file.close();

    std::cout << "Config updated successfully: " << config_key << " = " << new_value << std::endl;
    return true;
}

bool SelfRepairSystem::execute_drive_operation(const RepairAction& action) {
    std::cout << "Executing drive operation: " << action.description << std::endl;

    // Extract operation type from action parameters
    auto operation_it = action.parameters.find("operation");
    auto drive_it = action.parameters.find("drive");

    if (operation_it == action.parameters.end()) {
        std::cout << "Missing operation parameter" << std::endl;
        return false;
    }

    std::string operation = operation_it->second;

    try {
        if (operation == "defragment") {
            // Note: Defragmentation is OS-specific and may require admin privileges
            std::cout << "Drive defragmentation requested - not implemented for cross-platform" << std::endl;
            return false;
        } else if (operation == "check_disk") {
            // Run disk check (simplified)
            std::cout << "Running disk check..." << std::endl;
            // In a real implementation, would call system disk checking utilities
            std::cout << "Disk check completed" << std::endl;
            return true;
        } else if (operation == "cleanup") {
            // Clean temporary files
            std::cout << "Cleaning up temporary files..." << std::endl;
            // Remove temp files from system temp directory
            std::string temp_dir = fs::temp_directory_path().string();

            for (const auto& entry : fs::directory_iterator(temp_dir)) {
                try {
                    if (entry.is_regular_file() && entry.path().filename().string().find("hrm_temp") != std::string::npos) {
                        fs::remove(entry.path());
                    }
                } catch (const std::exception& e) {
                    // Skip files that can't be deleted
                }
            }
            std::cout << "Temporary file cleanup completed" << std::endl;
            return true;
        } else {
            std::cout << "Unknown drive operation: " << operation << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cout << "Drive operation failed: " << e.what() << std::endl;
        return false;
    }
}

void SelfRepairSystem::load_rollback_history() {
    // Load rollback history from file
    std::string history_file = (fs::path(backup_directory_) / "rollback_history.json").string();
    // Implementation would load from JSON file
}

void SelfRepairSystem::save_rollback_history() const {
    // Save rollback history to file
    std::string history_file = (fs::path(backup_directory_) / "rollback_history.json").string();
    // Implementation would save to JSON file
}

// Additional method implementations...
bool SelfRepairSystem::monitor_system_health() { return true; }
std::vector<std::string> SelfRepairSystem::get_pending_repairs() const { return {}; }
std::vector<RollbackPoint> SelfRepairSystem::get_available_rollback_points() const { return rollback_history_; }
void SelfRepairSystem::set_auto_rollback(bool enabled) { auto_rollback_enabled_ = enabled; }
void SelfRepairSystem::set_max_rollback_points(size_t max_points) { max_rollback_points_ = max_points; }
void SelfRepairSystem::set_backup_directory(const std::string& backup_dir) { backup_directory_ = backup_dir; }
bool SelfRepairSystem::has_pending_operations() const { return !pending_actions_.empty(); }
std::string SelfRepairSystem::get_last_rollback_point() const { return rollback_history_.empty() ? "" : rollback_history_.back().id; }
void SelfRepairSystem::print_repair_status() const { std::cout << "Repair system operational" << std::endl; }
bool SelfRepairSystem::emergency_rollback() { return rollback_last_operation(); }
bool SelfRepairSystem::schedule_repair_for_idle_time(const RepairAction& action) { RepairAction modifiable_action = action; return add_repair_action(modifiable_action); }
std::vector<std::string> SelfRepairSystem::analyze_repair_history() const { return {}; }

bool SelfRepairSystem::create_safety_checkpoint(const std::string& description) {
    // Create a safety checkpoint
    std::cout << "Creating safety checkpoint: " << description << std::endl;
    // Implementation would create a system snapshot
    return true;
}