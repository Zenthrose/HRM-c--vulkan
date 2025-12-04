#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <filesystem>

namespace fs = std::filesystem;

enum class SystemType {
    LINUX,
    WINDOWS,
    MACOS,
    UNKNOWN
};

enum class ProgramAccessLevel {
    USER,      // Can access user programs and directories
    SYSTEM,    // Can access system programs (requires elevated privileges)
    FULL       // Full system access (dangerous, not recommended)
};

struct SystemProgram {
    std::string name;
    std::string path;
    std::string description;
    std::vector<std::string> capabilities;
    bool requires_admin;
};

struct SystemDirectory {
    std::string name;
    std::string path;
    std::string description;
    ProgramAccessLevel required_access;
    bool is_safe; // Whether it's safe for the AI to access
};

struct AutoStartConfig {
    bool enable_auto_start;
    std::string application_name;
    std::string application_path;
    std::string description;
    bool start_minimized;
    bool start_with_system;
    std::vector<std::string> startup_arguments;
};

struct ProgramExecutionResult {
    bool success;
    int exit_code;
    std::string stdout_output;
    std::string stderr_output;
    std::chrono::milliseconds execution_time;
    std::string error_message;
};

class SystemIntegrationManager {
public:
    SystemIntegrationManager();
    ~SystemIntegrationManager();

    // System detection
    SystemType get_system_type() const;
    std::string get_system_info() const;
    std::unordered_map<std::string, std::string> get_environment_info() const;

    // Auto-start functionality
    bool setup_auto_start(const AutoStartConfig& config);
    bool remove_auto_start(const std::string& application_name);
    bool is_auto_start_enabled(const std::string& application_name) const;
    std::vector<AutoStartConfig> get_auto_start_entries() const;

    // Program discovery and execution
    std::vector<SystemProgram> discover_system_programs(ProgramAccessLevel access_level = ProgramAccessLevel::USER);
    std::vector<SystemDirectory> get_system_directories(ProgramAccessLevel access_level = ProgramAccessLevel::USER);
    ProgramExecutionResult execute_program(const std::string& program_path,
                                         const std::vector<std::string>& arguments = {},
                                         std::chrono::seconds timeout = std::chrono::seconds(30));

    // Safe system access
    bool is_path_safe(const std::string& path) const;
    bool validate_program_access(const std::string& program_path, ProgramAccessLevel required_access) const;
    std::vector<std::string> get_safe_programs() const;

    // File system operations (safe)
    std::vector<std::string> list_directory_safe(const std::string& path);
    bool file_exists_safe(const std::string& path) const;
    std::string read_text_file_safe(const std::string& path, size_t max_size = 1024 * 1024); // 1MB limit

    // System monitoring integration
    std::unordered_map<std::string, std::string> get_system_health_status() const;
    std::vector<std::string> get_system_warnings() const;

    // Safe command execution
    ProgramExecutionResult execute_safe_command(const std::string& command,
                                              std::chrono::seconds timeout = std::chrono::seconds(10));

private:
    SystemType system_type_;
    std::string home_directory_;
    std::string system_root_;
    std::vector<std::string> safe_paths_;
    std::vector<std::string> dangerous_commands_;

    // Platform-specific implementations
    bool setup_auto_start_linux(const AutoStartConfig& config);
    bool setup_auto_start_windows(const AutoStartConfig& config);
    bool setup_auto_start_macos(const AutoStartConfig& config);

    bool remove_auto_start_linux(const std::string& application_name);
    bool remove_auto_start_windows(const std::string& application_name);
    bool remove_auto_start_macos(const std::string& application_name);

    std::vector<SystemProgram> discover_programs_linux(ProgramAccessLevel access_level);
    std::vector<SystemProgram> discover_programs_windows(ProgramAccessLevel access_level);
    std::vector<SystemProgram> discover_programs_macos(ProgramAccessLevel access_level);

    ProgramExecutionResult execute_program_linux(const std::string& program_path,
                                               const std::vector<std::string>& arguments,
                                               std::chrono::seconds timeout);
    ProgramExecutionResult execute_program_windows(const std::string& program_path,
                                                 const std::vector<std::string>& arguments,
                                                 std::chrono::seconds timeout);
    ProgramExecutionResult execute_program_macos(const std::string& program_path,
                                               const std::vector<std::string>& arguments,
                                               std::chrono::seconds timeout);

    // Safety checks
    void initialize_safety_rules();
    bool is_command_safe(const std::string& command) const;
    bool contains_dangerous_patterns(const std::string& input) const;
    std::string sanitize_path(const std::string& path) const;

    // Helper functions
    SystemType detect_system_type();
    std::string get_home_directory();
    std::string get_system_root();
    std::string build_command_string(const std::string& program_path, const std::vector<std::string>& arguments);
    ProgramExecutionResult parse_execution_output(const std::string& output, int exit_code,
                                                std::chrono::milliseconds duration);
};