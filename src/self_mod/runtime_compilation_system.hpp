#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <filesystem>

namespace fs = std::filesystem;

struct CompilationResult {
    bool success;
    std::string executable_path;
    std::string library_path;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    double compilation_time_seconds;
};

struct RuntimeModule {
    void* handle;
    std::string module_path;
    std::unordered_map<std::string, void*> symbols;

    RuntimeModule() : handle(nullptr) {}
    ~RuntimeModule() {
        if (handle) {
            // dlclose(handle); // Would need proper cleanup
        }
    }
};

class RuntimeCompilationSystem {
public:
    RuntimeCompilationSystem(const std::string& temp_dir = "/tmp/hrm_compilation");
    ~RuntimeCompilationSystem();

    // Compilation methods
    CompilationResult compile_to_executable(const std::vector<std::string>& source_files,
                                          const std::string& output_name,
                                          const std::vector<std::string>& include_dirs = {},
                                          const std::vector<std::string>& link_libraries = {});

    CompilationResult compile_to_library(const std::vector<std::string>& source_files,
                                       const std::string& output_name,
                                       const std::vector<std::string>& include_dirs = {},
                                       const std::vector<std::string>& link_libraries = {});

    // Module loading and hot-swapping
    std::shared_ptr<RuntimeModule> load_module(const std::string& library_path);
    bool unload_module(std::shared_ptr<RuntimeModule> module);
    bool hot_swap_module(const std::string& old_library_path, const std::string& new_library_path);

    // Code modification and recompilation
    CompilationResult modify_and_recompile(const std::string& source_file,
                                         const std::string& modification_script,
                                         const std::string& output_name);

    // Safety and validation
    bool validate_compilation(const CompilationResult& result);
    bool test_module_compatibility(const std::string& library_path);
    void create_backup(const std::string& file_path);
    bool restore_backup(const std::string& file_path);

    // System management
    void cleanup_temp_files();
    std::unordered_map<std::string, std::string> get_system_info();

private:
    std::string temp_directory_;
    std::vector<std::string> active_modules_;
    std::unordered_map<std::string, std::string> backup_files_;

    // Compilation helpers
    std::string generate_compile_command(const std::vector<std::string>& source_files,
                                       const std::string& output_path,
                                       bool is_library,
                                       const std::vector<std::string>& include_dirs,
                                       const std::vector<std::string>& link_libraries);

    CompilationResult execute_compilation(const std::string& command,
                                        const std::vector<std::string>& source_files);

    // File system helpers
    bool create_temp_directory();
    std::string get_unique_temp_path(const std::string& prefix);
    bool copy_file(const std::string& src, const std::string& dst);

    // Module management
    void* load_library(const std::string& path);
    void unload_library(void* handle);
    void* get_symbol(void* handle, const std::string& symbol_name);

    // Safety checks
    bool check_compilation_safety(const std::vector<std::string>& source_files);
    bool validate_source_code(const std::string& source_file);
    std::vector<std::string> detect_dangerous_patterns(const std::string& code);

    // Error handling
    std::string format_compilation_errors(const std::string& error_output);
    std::pair<std::vector<std::string>, std::vector<std::string>> parse_error_output(const std::string& error_output);
};