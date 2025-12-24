#include "runtime_compilation_system.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;
#ifdef _WIN32
#include <windows.h>
#include <process.h>
#include <io.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#include <sys/wait.h>
#endif
#include <cstring>

RuntimeCompilationSystem::RuntimeCompilationSystem(const std::string& temp_dir)
    : temp_directory_(temp_dir) {
    std::cout << "Initializing Runtime Compilation System..." << std::endl;

    if (!create_temp_directory()) {
        std::cerr << "Warning: Could not create temp directory: " << temp_directory_ << std::endl;
    }

    std::cout << "Runtime Compilation System initialized" << std::endl;
}

RuntimeCompilationSystem::~RuntimeCompilationSystem() {
    cleanup_temp_files();
}

CompilationResult RuntimeCompilationSystem::compile_to_executable(
    const std::vector<std::string>& source_files,
    const std::string& output_name,
    const std::vector<std::string>& include_dirs,
    const std::vector<std::string>& link_libraries) {

    std::string output_path = temp_directory_ + "/" + output_name;
    std::string command = generate_compile_command(source_files, output_path, false,
                                                 include_dirs, link_libraries);

    return execute_compilation(command, source_files);
}

CompilationResult RuntimeCompilationSystem::compile_to_library(
    const std::vector<std::string>& source_files,
    const std::string& output_name,
    const std::vector<std::string>& include_dirs,
    const std::vector<std::string>& link_libraries) {

    std::string output_path = temp_directory_ + "/" + output_name;
    std::string command = generate_compile_command(source_files, output_path, true,
                                                 include_dirs, link_libraries);

    return execute_compilation(command, source_files);
}

std::shared_ptr<RuntimeModule> RuntimeCompilationSystem::load_module(const std::string& library_path) {
    auto module = std::make_shared<RuntimeModule>();
    module->module_path = library_path;
    module->handle = load_library(library_path);

    if (!module->handle) {
        std::cerr << "Failed to load module: " << library_path << std::endl;
        return nullptr;
    }

    active_modules_.push_back(library_path);
    return module;
}

bool RuntimeCompilationSystem::unload_module(std::shared_ptr<RuntimeModule> module) {
    if (!module || !module->handle) return false;

    unload_library(module->handle);
    module->handle = nullptr;

    // Remove from active modules
    auto it = std::find(active_modules_.begin(), active_modules_.end(), module->module_path);
    if (it != active_modules_.end()) {
        active_modules_.erase(it);
    }

    return true;
}

bool RuntimeCompilationSystem::hot_swap_module(const std::string& old_library_path,
                                             const std::string& new_library_path) {
    // Find the old module
    auto it = std::find(active_modules_.begin(), active_modules_.end(), old_library_path);
    if (it == active_modules_.end()) {
        std::cerr << "Old module not found in active modules" << std::endl;
        return false;
    }

    // Load new module
    void* new_handle = load_library(new_library_path);
    if (!new_handle) {
        std::cerr << "Failed to load new module" << std::endl;
        return false;
    }

    // Unload old module
    void* old_handle = nullptr;
    // In a real implementation, we'd need to track handles
    // For now, just update the active modules list

    *it = new_library_path;

    std::cout << "Hot-swapped module: " << old_library_path << " -> " << new_library_path << std::endl;
    return true;
}

CompilationResult RuntimeCompilationSystem::modify_and_recompile(
    const std::string& source_file,
    const std::string& modification_script,
    const std::string& output_name) {

    CompilationResult result;

    try {
        // 1. Read the source file
        std::ifstream input_file(source_file);
        if (!input_file.is_open()) {
            result.success = false;
            result.errors = {"Failed to open source file: " + source_file};
            return result;
        }

        std::string source_content((std::istreambuf_iterator<char>(input_file)),
                                   std::istreambuf_iterator<char>());
        input_file.close();

        // 2. Parse modification script (simple format: "old_text|new_text")
        size_t separator_pos = modification_script.find('|');
        if (separator_pos == std::string::npos) {
            result.success = false;
            result.errors = {"Invalid modification script format. Expected: 'old_text|new_text'"};
            return result;
        }

        std::string old_text = modification_script.substr(0, separator_pos);
        std::string new_text = modification_script.substr(separator_pos + 1);

        // 3. Apply modification
        size_t pos = source_content.find(old_text);
        if (pos == std::string::npos) {
            result.success = false;
            result.errors = {"Old text not found in source file"};
            return result;
        }

        source_content.replace(pos, old_text.length(), new_text);

        // 4. Write modified source to temporary file
        std::string temp_source_file = source_file + ".modified";
        std::ofstream output_file(temp_source_file);
        if (!output_file.is_open()) {
            result.success = false;
            result.errors = {"Failed to create modified source file"};
            return result;
        }
        output_file << source_content;
        output_file.close();

        // 5. Compile the modified source
        std::vector<std::string> source_files = {temp_source_file};
#ifdef _WIN32
        std::string output_path = output_name + ".dll";
#else
        std::string output_path = output_name + ".so";
#endif

        result = compile_to_library(source_files, output_path);

        // 6. Clean up temporary file
        fs::remove(temp_source_file);

        if (result.success) {
            std::cout << "Successfully modified and recompiled: " << source_file << std::endl;
        }

    } catch (const std::exception& e) {
        result.success = false;
        result.errors = {"Exception during modify_and_recompile: " + std::string(e.what())};
    }

    return result;
}

bool RuntimeCompilationSystem::validate_compilation(const CompilationResult& result) {
    if (!result.success) return false;

    // Check if output files exist
    if (!result.executable_path.empty() && !fs::exists(result.executable_path)) return false;
    if (!result.library_path.empty() && !fs::exists(result.library_path)) return false;

    // Check for critical errors
    for (const auto& error : result.errors) {
        if (error.find("error") != std::string::npos ||
            error.find("Error") != std::string::npos) {
            return false;
        }
    }

    return true;
}

bool RuntimeCompilationSystem::test_module_compatibility(const std::string& library_path) {
    // Load the module temporarily to test compatibility
    void* handle = load_library(library_path);
    if (!handle) return false;

    // Try to find key symbols (this would be application-specific)
    // For HRM, we might check for specific function symbols

    unload_library(handle);
    return true;
}

void RuntimeCompilationSystem::create_backup(const std::string& file_path) {
    std::string backup_path = file_path + ".backup";
    if (copy_file(file_path, backup_path)) {
        backup_files_[file_path] = backup_path;
    }
}

bool RuntimeCompilationSystem::restore_backup(const std::string& file_path) {
    auto it = backup_files_.find(file_path);
    if (it == backup_files_.end()) return false;

    bool success = copy_file(it->second, file_path);
    if (success) {
        fs::remove(it->second);
        backup_files_.erase(it);
    }

    return success;
}

void RuntimeCompilationSystem::cleanup_temp_files() {
    try {
        if (fs::exists(temp_directory_)) {
            fs::remove_all(temp_directory_);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error cleaning up temp files: " << e.what() << std::endl;
    }

    // Clean up backup files
    for (const auto& pair : backup_files_) {
        try {
            if (fs::exists(pair.second)) {
                fs::remove(pair.second);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error removing backup file: " << e.what() << std::endl;
        }
    }
    backup_files_.clear();
}

std::unordered_map<std::string, std::string> RuntimeCompilationSystem::get_system_info() {
    std::unordered_map<std::string, std::string> info;

    info["temp_directory"] = temp_directory_;
    info["active_modules_count"] = std::to_string(active_modules_.size());
    info["backup_files_count"] = std::to_string(backup_files_.size());

    // Check compiler availability
    int gpp_result = system("g++ --version > /dev/null 2>&1");
    info["g++_available"] = (gpp_result == 0) ? "true" : "false";

    int clang_result = system("clang++ --version > /dev/null 2>&1");
    info["clang++_available"] = (clang_result == 0) ? "true" : "false";

    return info;
}

// Private methods

std::string RuntimeCompilationSystem::generate_compile_command(
    const std::vector<std::string>& source_files,
    const std::string& output_path,
    bool is_library,
    const std::vector<std::string>& include_dirs,
    const std::vector<std::string>& link_libraries) {

    std::string command = "g++ -std=c++17 -O2 -Wall -Wextra";

    // Add include directories
    for (const auto& include_dir : include_dirs) {
        command += " -I" + include_dir;
    }

    // Add source files
    for (const auto& source_file : source_files) {
        command += " " + source_file;
    }

    // Add link libraries
    for (const auto& lib : link_libraries) {
        command += " -l" + lib;
    }

    // Output options
    if (is_library) {
        command += " -shared -fPIC -o " + output_path + ".so";
    } else {
        command += " -o " + output_path;
    }

    return command;
}

CompilationResult RuntimeCompilationSystem::execute_compilation(
    const std::string& command,
    const std::vector<std::string>& source_files) {

    CompilationResult result;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "Executing compilation command: " << command << std::endl;

    // Execute the command
#ifdef _WIN32
    FILE* pipe = _popen(command.c_str(), "r");
#else
    FILE* pipe = popen(command.c_str(), "r");
#endif
    if (!pipe) {
        result.success = false;
        result.errors = {"Failed to execute compilation command"};
        return result;
    }

    // Read output
    char buffer[4096];
    std::string output;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }

#ifdef _WIN32
    int status = _pclose(pipe);
#else
    int status = pclose(pipe);
#endif
    result.success = (status == 0);

    auto end_time = std::chrono::high_resolution_clock::now();
    result.compilation_time_seconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;

    // Parse output for errors and warnings
    auto parsed_output = parse_error_output(output);
    result.errors = parsed_output.first;
    result.warnings = parsed_output.second;

    // Set output paths
    if (command.find("-shared") != std::string::npos) {
        size_t pos = command.find("-o ");
        if (pos != std::string::npos) {
            result.library_path = command.substr(pos + 3);
            // Remove trailing spaces and newlines
            size_t end_pos = result.library_path.find_first_of(" \t\n\r\f\v");
            if (end_pos != std::string::npos) {
                result.library_path = result.library_path.substr(0, end_pos);
            }
        }
    } else {
        size_t pos = command.find("-o ");
        if (pos != std::string::npos) {
            result.executable_path = command.substr(pos + 3);
            size_t end_pos = result.executable_path.find_first_of(" \t\n\r\f\v");
            if (end_pos != std::string::npos) {
                result.executable_path = result.executable_path.substr(0, end_pos);
            }
        }
    }

    return result;
}

bool RuntimeCompilationSystem::create_temp_directory() {
    try {
        // Create parent directories if needed
        fs::path temp_path(temp_directory_);
        fs::path parent = temp_path.parent_path();

        if (!parent.empty() && !fs::exists(parent)) {
            fs::create_directories(parent);
        }

        // Create the temp directory, but don't fail if it already exists
        std::error_code ec;
        fs::create_directories(temp_directory_, ec);
        if (ec) {
            std::cerr << "Error creating temp directory: " << ec.message() << std::endl;
            return false;
        }

        // Check if directory exists and is writable
        return fs::exists(temp_directory_) && fs::is_directory(temp_directory_);
    } catch (const std::exception& e) {
        std::cerr << "Error creating temp directory: " << e.what() << std::endl;
        
        // Fallback: try current directory
        temp_directory_ = "./temp";
        try {
            return fs::create_directories(temp_directory_);
        } catch (...) {
            temp_directory_ = ".";
            return true; // Use current directory as fallback
        }
    }
}

std::string RuntimeCompilationSystem::get_unique_temp_path(const std::string& prefix) {
    static int counter = 0;
    return temp_directory_ + "/" + prefix + "_" + std::to_string(counter++);
}

bool RuntimeCompilationSystem::copy_file(const std::string& src, const std::string& dst) {
    try {
        fs::copy_file(src, dst, fs::copy_options::overwrite_existing);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error copying file: " << e.what() << std::endl;
        return false;
    }
}

void* RuntimeCompilationSystem::load_library(const std::string& path) {
#ifdef _WIN32
    return LoadLibraryA(path.c_str());
#else
    return dlopen(path.c_str(), RTLD_LAZY);
#endif
}

void RuntimeCompilationSystem::unload_library(void* handle) {
    if (handle) {
#ifdef _WIN32
        FreeLibrary(static_cast<HMODULE>(handle));
#else
        dlclose(handle);
#endif
    }
}

void* RuntimeCompilationSystem::get_symbol(void* handle, const std::string& symbol_name) {
    if (!handle) return nullptr;
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), symbol_name.c_str()));
#else
    return dlsym(handle, symbol_name.c_str());
#endif
}

bool RuntimeCompilationSystem::check_compilation_safety(const std::vector<std::string>& source_files) {
    for (const auto& file : source_files) {
        if (!validate_source_code(file)) {
            return false;
        }
    }
    return true;
}

bool RuntimeCompilationSystem::validate_source_code(const std::string& source_file) {
    std::ifstream file(source_file);
    if (!file.is_open()) return false;

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string code = buffer.str();

    auto dangerous_patterns = detect_dangerous_patterns(code);
    return dangerous_patterns.empty();
}

std::vector<std::string> RuntimeCompilationSystem::detect_dangerous_patterns(const std::string& code) {
    std::vector<std::string> dangerous;

    // Check for system calls that could be harmful
    if (code.find("system(") != std::string::npos) {
        dangerous.push_back("Use of system() calls detected");
    }

    if (code.find("exec") != std::string::npos) {
        dangerous.push_back("Use of exec functions detected");
    }

    if (code.find("fork") != std::string::npos) {
        dangerous.push_back("Use of fork() detected");
    }

    return dangerous;
}

std::string RuntimeCompilationSystem::format_compilation_errors(const std::string& error_output) {
    // Parse and format compiler errors into a human-readable format
    std::stringstream formatted;
    std::stringstream ss(error_output);
    std::string line;
    
    formatted << "=== Compilation Errors ===\n";
    
    int error_count = 0;
    int warning_count = 0;
    
    while (std::getline(ss, line)) {
        if (line.empty()) continue;
        
        // Categorize and format each line
        if (line.find("error:") != std::string::npos) {
            error_count++;
            formatted << "[ERROR] " << line << "\n";
        } else if (line.find("warning:") != std::string::npos) {
            warning_count++;
            formatted << "[WARN ] " << line << "\n";
        } else if (line.find("note:") != std::string::npos) {
            formatted << "[INFO ] " << line << "\n";
        } else {
            formatted << "        " << line << "\n";
        }
    }
    
    formatted << "\n=== Summary ===\n";
    formatted << "Errors: " << error_count << ", Warnings: " << warning_count << "\n";
    
    return formatted.str();
}

std::pair<std::vector<std::string>, std::vector<std::string>>
RuntimeCompilationSystem::parse_error_output(const std::string& error_output) {
    std::vector<std::string> errors;
    std::vector<std::string> warnings;

    std::stringstream ss(error_output);
    std::string line;

    while (std::getline(ss, line)) {
        if (line.find("error") != std::string::npos ||
            line.find("Error") != std::string::npos) {
            errors.push_back(line);
        } else if (line.find("warning") != std::string::npos ||
                   line.find("Warning") != std::string::npos) {
            warnings.push_back(line);
        }
    }

    return {errors, warnings};
}