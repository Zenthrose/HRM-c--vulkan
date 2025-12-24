#include "system_integration_manager.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#ifdef _WIN32
#include <windows.h>
#include <process.h>
#include <io.h>
#include <lmcons.h>  // For UNLEN
#include <shlobj.h>  // For SHGetFolderPath
#else
#include <unistd.h>
#include <sys/wait.h>
#include <pwd.h>
#include <limits.h>
#include <dirent.h>
#include <sys/statvfs.h>
#endif
#include <algorithm>

SystemIntegrationManager::SystemIntegrationManager()
    : system_type_(detect_system_type()) {

    home_directory_ = get_home_directory();
    system_root_ = get_system_root();
    initialize_safety_rules();

    std::cout << "System Integration Manager initialized for "
              << (system_type_ == SystemType::LINUX ? "Linux" :
                  system_type_ == SystemType::WINDOWS ? "Windows" :
                  system_type_ == SystemType::MACOS ? "macOS" : "Unknown")
              << " system" << std::endl;
}

SystemIntegrationManager::~SystemIntegrationManager() {
    // Cleanup any temporary resources
}

SystemType SystemIntegrationManager::get_system_type() const {
    return system_type_;
}

std::string SystemIntegrationManager::get_system_info() const {
    std::stringstream ss;

#ifdef __linux__
    ss << "Linux";
#elif _WIN32
    ss << "Windows";
#elif __APPLE__
    ss << "macOS";
#else
    ss << "Unknown";
#endif

    // Add architecture info
#ifdef __x86_64__
    ss << " x86_64";
#elif __aarch64__
    ss << " ARM64";
#endif

    return ss.str();
}

std::unordered_map<std::string, std::string> SystemIntegrationManager::get_environment_info() const {
    std::unordered_map<std::string, std::string> env_info;

    // Get common environment variables
    const char* env_vars[] = {
        "HOME", "USER", "PATH", "SHELL", "LANG", "DISPLAY",
        "XDG_SESSION_TYPE", "DESKTOP_SESSION", nullptr
    };

    for (int i = 0; env_vars[i] != nullptr; ++i) {
        const char* value = std::getenv(env_vars[i]);
        if (value) {
            env_info[env_vars[i]] = value;
        }
    }

    // Add system-specific info
    env_info["SYSTEM_TYPE"] = get_system_info();
    env_info["HOME_DIR"] = home_directory_;
    env_info["SYSTEM_ROOT"] = system_root_;

    return env_info;
}

bool SystemIntegrationManager::setup_auto_start(const AutoStartConfig& config) {
    switch (system_type_) {
        case SystemType::LINUX:
            return setup_auto_start_linux(config);
        case SystemType::WINDOWS:
            return setup_auto_start_windows(config);
        case SystemType::MACOS:
            return setup_auto_start_macos(config);
        default:
            std::cerr << "Auto-start not supported on this system type" << std::endl;
            return false;
    }
}

bool SystemIntegrationManager::remove_auto_start(const std::string& application_name) {
    switch (system_type_) {
        case SystemType::LINUX:
            return remove_auto_start_linux(application_name);
        case SystemType::WINDOWS:
            return remove_auto_start_windows(application_name);
        case SystemType::MACOS:
            return remove_auto_start_macos(application_name);
        default:
            return false;
    }
}

bool SystemIntegrationManager::is_auto_start_enabled(const std::string& application_name) const {
    auto entries = get_auto_start_entries();
    return std::any_of(entries.begin(), entries.end(),
                      [&application_name](const AutoStartConfig& entry) {
                          return entry.application_name == application_name;
                      });
}

std::vector<AutoStartConfig> SystemIntegrationManager::get_auto_start_entries() const {
    std::vector<AutoStartConfig> entries;

    switch (system_type_) {
        case SystemType::LINUX: {
            // Check ~/.config/autostart/
            std::string autostart_dir = home_directory_ + "/.config/autostart/";
            if (fs::exists(autostart_dir)) {
                for (const auto& entry : fs::directory_iterator(autostart_dir)) {
                    if (entry.path().extension() == ".desktop") {
                        // Parse .desktop file (simplified)
                        std::ifstream file(entry.path());
                        std::string line;
                        AutoStartConfig config;
                        config.application_name = entry.path().stem().string();

                        while (std::getline(file, line)) {
                            if (line.find("Exec=") == 0) {
                                config.application_path = line.substr(5);
                            } else if (line.find("Comment=") == 0) {
                                config.description = line.substr(8);
                            }
                        }

                        if (!config.application_path.empty()) {
                            entries.push_back(config);
                        }
                    }
                }
            }
            break;
        }
        // Windows and macOS implementations would go here
        default:
            break;
    }

    return entries;
}

std::vector<SystemProgram> SystemIntegrationManager::discover_system_programs(ProgramAccessLevel access_level) {
    switch (system_type_) {
        case SystemType::LINUX:
            return discover_programs_linux(access_level);
        case SystemType::WINDOWS:
            return discover_programs_windows(access_level);
        case SystemType::MACOS:
            return discover_programs_macos(access_level);
        default:
            return {};
    }
}

std::vector<SystemDirectory> SystemIntegrationManager::get_system_directories(ProgramAccessLevel access_level) {
    std::vector<SystemDirectory> directories;

    // Safe user directories
    if (access_level >= ProgramAccessLevel::USER) {
        directories.push_back({
            "Home", home_directory_, "User home directory", ProgramAccessLevel::USER, true
        });
        directories.push_back({
            "Documents", home_directory_ + "/Documents", "User documents", ProgramAccessLevel::USER, true
        });
        directories.push_back({
            "Downloads", home_directory_ + "/Downloads", "User downloads", ProgramAccessLevel::USER, true
        });
        directories.push_back({
            "Desktop", home_directory_ + "/Desktop", "User desktop", ProgramAccessLevel::USER, true
        });
    }

    // System directories (more restricted)
    if (access_level >= ProgramAccessLevel::SYSTEM) {
        directories.push_back({
            "System Bin", "/usr/bin", "System binary directory", ProgramAccessLevel::SYSTEM, false
        });
        directories.push_back({
            "Local Bin", "/usr/local/bin", "Local binary directory", ProgramAccessLevel::SYSTEM, false
        });
    }

    return directories;
}

ProgramExecutionResult SystemIntegrationManager::execute_program(const std::string& program_path,
                                                               const std::vector<std::string>& arguments,
                                                               std::chrono::seconds timeout) {
    // Validate safety first
    if (!is_path_safe(program_path)) {
        return {false, -1, "", "Unsafe path", std::chrono::milliseconds(0), "Path validation failed"};
    }

    if (!validate_program_access(program_path, ProgramAccessLevel::USER)) {
        return {false, -1, "", "Access denied", std::chrono::milliseconds(0), "Permission denied"};
    }

    switch (system_type_) {
        case SystemType::LINUX:
            return execute_program_linux(program_path, arguments, timeout);
        case SystemType::WINDOWS:
            return execute_program_windows(program_path, arguments, timeout);
        case SystemType::MACOS:
            return execute_program_macos(program_path, arguments, timeout);
        default:
            return {false, -1, "", "", std::chrono::milliseconds(0), "Unsupported system type"};
    }
}

bool SystemIntegrationManager::is_path_safe(const std::string& path) const {
    // Check against dangerous paths
    std::vector<std::string> dangerous_paths = {
        "/bin", "/sbin", "/usr/sbin", "/etc", "/root", "/sys", "/proc", "/dev"
    };

    for (const auto& dangerous : dangerous_paths) {
        if (path.find(dangerous) == 0) {
            return false;
        }
    }

    // Check if path is within allowed directories
    for (const auto& safe_path : safe_paths_) {
        if (path.find(safe_path) == 0) {
            return true;
        }
    }

    return false;
}

bool SystemIntegrationManager::validate_program_access(const std::string& program_path,
                                                     ProgramAccessLevel required_access) const {
    // For now, only allow user-level access
    // In a real implementation, this would check file permissions
    return required_access <= ProgramAccessLevel::USER;
}

std::vector<std::string> SystemIntegrationManager::get_safe_programs() const {
    std::vector<std::string> safe_programs = {
        "ls", "cat", "grep", "find", "which", "echo", "pwd",
        "date", "cal", "df", "du", "free", "ps", "top", "htop"
    };

    return safe_programs;
}

std::vector<std::string> SystemIntegrationManager::list_directory_safe(const std::string& path) {
    std::vector<std::string> contents;

    if (!is_path_safe(path)) {
        return contents;
    }

#ifdef _WIN32
    WIN32_FIND_DATAA findData;
    HANDLE hFind = FindFirstFileA((path + "\\*").c_str(), &findData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            std::string name = findData.cFileName;
            if (name != "." && name != "..") {
                contents.push_back(name);
            }
        } while (FindNextFileA(hFind, &findData));
        FindClose(hFind);
    }
#else
    DIR* dir = opendir(path.c_str());
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string name = entry->d_name;
            if (name != "." && name != "..") {
                contents.push_back(name);
            }
        }
        closedir(dir);
    }
#endif

    return contents;
}

bool SystemIntegrationManager::file_exists_safe(const std::string& path) const {
    if (!is_path_safe(path)) {
        return false;
    }

    return fs::exists(path);
}

std::string SystemIntegrationManager::read_text_file_safe(const std::string& path, size_t max_size) {
    if (!is_path_safe(path) || !fs::exists(path) || !fs::is_regular_file(path)) {
        return "";
    }

    // Check file size
    auto file_size = fs::file_size(path);
    if (file_size > max_size) {
        return "[File too large to read safely]";
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        return "";
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());

    return content;
}

std::unordered_map<std::string, std::string> SystemIntegrationManager::get_system_health_status() const {
    std::unordered_map<std::string, std::string> status;

    // Basic system health checks
    status["system_type"] = get_system_info();

    // Check disk space
#ifdef _WIN32
    ULARGE_INTEGER freeBytesAvailable, totalNumberOfBytes, totalNumberOfFreeBytes;
    if (GetDiskFreeSpaceExA(home_directory_.c_str(), &freeBytesAvailable, &totalNumberOfBytes, &totalNumberOfFreeBytes)) {
        double available_gb = freeBytesAvailable.QuadPart / (1024.0 * 1024.0 * 1024.0);
        status["disk_space_available_gb"] = std::to_string(available_gb);
        if (available_gb < 1.0) {
            status["disk_warning"] = "Low disk space";
        }
    }
#else
    struct statvfs stat;
    if (statvfs(home_directory_.c_str(), &stat) == 0) {
        uint64_t available_bytes = stat.f_bavail * stat.f_frsize;
        double available_gb = available_bytes / (1024.0 * 1024.0 * 1024.0);
        status["disk_space_available_gb"] = std::to_string(available_gb);
        if (available_gb < 1.0) {
            status["disk_warning"] = "Low disk space";
        }
    }
#endif

    // Check memory
#ifdef _WIN32
    MEMORYSTATUSEX memStatus;
    memStatus.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memStatus)) {
        double available_mb = memStatus.ullAvailPhys / (1024.0 * 1024.0);
        status["memory_available_mb"] = std::to_string(available_mb);
        if (available_mb < 512.0) {
            status["memory_warning"] = "Low memory";
        }
    }
#else
    std::ifstream meminfo("/proc/meminfo");
    if (meminfo.is_open()) {
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.find("MemAvailable:") == 0) {
                std::istringstream iss(line);
                std::string label;
                uint64_t value;
                std::string unit;
                iss >> label >> value >> unit;

                double available_mb = value / 1024.0;
                status["memory_available_mb"] = std::to_string(available_mb);

                if (available_mb < 512.0) {
                    status["memory_warning"] = "Low memory";
                }
                break;
            }
        }
    }
#endif

    return status;
}

std::vector<std::string> SystemIntegrationManager::get_system_warnings() const {
    std::vector<std::string> warnings;
    auto health = get_system_health_status();

    if (health.find("disk_warning") != health.end()) {
        warnings.push_back(health["disk_warning"]);
    }

    if (health.find("memory_warning") != health.end()) {
        warnings.push_back(health["memory_warning"]);
    }

    return warnings;
}

ProgramExecutionResult SystemIntegrationManager::execute_safe_command(const std::string& command,
                                                                    std::chrono::seconds timeout) {
    if (!is_command_safe(command)) {
        return {false, -1, "", "", std::chrono::milliseconds(0), "Command not allowed"};
    }

    // For safe commands, allow execution with restrictions
    return execute_program("/bin/bash", {"-c", command}, timeout);
}

// Private methods

bool SystemIntegrationManager::setup_auto_start_linux(const AutoStartConfig& config) {
    std::string autostart_dir = home_directory_ + "/.config/autostart/";
    fs::create_directories(autostart_dir);

    std::string desktop_file = autostart_dir + config.application_name + ".desktop";

    std::ofstream file(desktop_file);
    if (!file.is_open()) {
        return false;
    }

    file << "[Desktop Entry]\n";
    file << "Type=Application\n";
    file << "Name=" << config.application_name << "\n";
    file << "Exec=" << config.application_path;

    if (!config.startup_arguments.empty()) {
        file << " " << config.startup_arguments[0]; // Simplified - only first arg
    }

    file << "\n";

    if (!config.description.empty()) {
        file << "Comment=" << config.description << "\n";
    }

    file << "Terminal=false\n";
    file << "StartupNotify=false\n";

    return true;
}

bool SystemIntegrationManager::setup_auto_start_windows(const AutoStartConfig& config) {
    // Windows implementation would use registry or startup folder
    std::cerr << "Windows auto-start not implemented" << std::endl;
    return false;
}

bool SystemIntegrationManager::setup_auto_start_macos(const AutoStartConfig& config) {
    // macOS implementation would use launch agents
    std::cerr << "macOS auto-start not implemented" << std::endl;
    return false;
}

bool SystemIntegrationManager::remove_auto_start_linux(const std::string& application_name) {
    std::string desktop_file = home_directory_ + "/.config/autostart/" + application_name + ".desktop";

    if (fs::exists(desktop_file)) {
        fs::remove(desktop_file);
        return true;
    }

    return false;
}

bool SystemIntegrationManager::remove_auto_start_windows(const std::string& application_name) {
    std::cerr << "Windows auto-start removal not implemented" << std::endl;
    return false;
}

bool SystemIntegrationManager::remove_auto_start_macos(const std::string& application_name) {
    std::cerr << "macOS auto-start removal not implemented" << std::endl;
    return false;
}

std::vector<SystemProgram> SystemIntegrationManager::discover_programs_linux(ProgramAccessLevel access_level) {
    std::vector<SystemProgram> programs;

    // Safe user programs
    if (access_level >= ProgramAccessLevel::USER) {
        std::vector<std::string> user_paths = {
            "/usr/bin", "/usr/local/bin", home_directory_ + "/bin"
        };

        for (const auto& path : user_paths) {
            if (fs::exists(path)) {
                for (const auto& entry : fs::directory_iterator(path)) {
                    if (fs::is_regular_file(entry) && (fs::status(entry).permissions() & fs::perms::owner_exec) != fs::perms::none) {
                        SystemProgram program;
                        program.name = entry.path().filename().string();
                        program.path = entry.path().string();
                        program.description = "System program";
                        program.requires_admin = false;

                        // Add some basic capabilities based on name
                        if (program.name.find("python") != std::string::npos) {
                            program.capabilities = {"scripting", "automation"};
                        } else if (program.name.find("git") != std::string::npos) {
                            program.capabilities = {"version_control", "collaboration"};
                        }

                        programs.push_back(program);
                    }
                }
            }
        }
    }

    return programs;
}

#ifdef _WIN32
std::vector<SystemProgram> SystemIntegrationManager::discover_programs_windows(ProgramAccessLevel access_level) {
    std::vector<SystemProgram> programs;

    // Scan Program Files directories
    std::vector<std::string> program_dirs = {
        "C:\\Program Files",
        "C:\\Program Files (x86)"
    };

    for (const auto& dir_path : program_dirs) {
        WIN32_FIND_DATAA findData;
        std::string search_path = dir_path + "\\*";

        HANDLE hFind = FindFirstFileA(search_path.c_str(), &findData);
        if (hFind != INVALID_HANDLE_VALUE) {
            do {
                if (findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                    std::string name = findData.cFileName;
                    if (name != "." && name != "..") {
                        // Look for .exe files in the directory
                        std::string exe_path = dir_path + "\\" + name + "\\" + name + ".exe";
                        if (GetFileAttributesA(exe_path.c_str()) != INVALID_FILE_ATTRIBUTES) {
                            SystemProgram program;
                            program.name = name;
                            program.path = exe_path;
                            program.description = "Installed program from " + dir_path;
                            program.requires_admin = (access_level == ProgramAccessLevel::SYSTEM);
                            programs.push_back(program);
                        }
                    }
                }
            } while (FindNextFileA(hFind, &findData));
            FindClose(hFind);
        }
    }

    return programs;
}
#endif

std::vector<SystemProgram> SystemIntegrationManager::discover_programs_macos(ProgramAccessLevel access_level) {
    // macOS implementation
    return {};
}

ProgramExecutionResult SystemIntegrationManager::execute_program_linux(const std::string& program_path,
                                                                     const std::vector<std::string>& arguments,
                                                                     std::chrono::seconds timeout) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Build command
    std::string command = build_command_string(program_path, arguments);

    // Execute with timeout using popen
#ifdef _WIN32
    FILE* pipe = _popen(command.c_str(), "r");
#else
    FILE* pipe = popen(command.c_str(), "r");
#endif
    if (!pipe) {
        return {false, -1, "", "", std::chrono::milliseconds(0), "Failed to execute command"};
    }

    // Read output with timeout (simplified)
    std::string output;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }

#ifdef _WIN32
    int exit_code = _pclose(pipe);
#else
    int exit_code = pclose(pipe);
#endif

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    return parse_execution_output(output, exit_code, duration);
}

#ifdef _WIN32
ProgramExecutionResult SystemIntegrationManager::execute_program_windows(const std::string& program_path,
                                                                       const std::vector<std::string>& arguments,
                                                                       std::chrono::seconds timeout) {
    // Windows implementation using CreateProcess
    std::string command = build_command_string(program_path, arguments);

    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    SECURITY_ATTRIBUTES sa;
    HANDLE hReadPipe, hWritePipe;

    ZeroMemory(&si, sizeof(si));
    ZeroMemory(&pi, sizeof(pi));
    ZeroMemory(&sa, sizeof(sa));

    sa.nLength = sizeof(SECURITY_ATTRIBUTES);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = NULL;

    // Create pipe for stdout
    if (!CreatePipe(&hReadPipe, &hWritePipe, &sa, 0)) {
        return {false, -1, "", "", std::chrono::milliseconds(0), "Failed to create pipe"};
    }

    si.cb = sizeof(STARTUPINFOA);
    si.hStdError = hWritePipe;
    si.hStdOutput = hWritePipe;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.dwFlags |= STARTF_USESTDHANDLES;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Create the process
    if (!CreateProcessA(NULL, const_cast<char*>(command.c_str()), NULL, NULL, TRUE,
                       0, NULL, NULL, &si, &pi)) {
        CloseHandle(hReadPipe);
        CloseHandle(hWritePipe);
        return {false, -1, "", "", std::chrono::milliseconds(0), "Failed to create process"};
    }

    // Close write end of pipe
    CloseHandle(hWritePipe);

    // Resource-aware timeout calculation
    DWORD adaptive_timeout_ms;
    if (timeout.count() == 0) {
        // Calculate adaptive timeout based on operation type
        adaptive_timeout_ms = 30000; // 30 seconds base
        if (program_path.find("compile") != std::string::npos) adaptive_timeout_ms = 120000; // 2 min for compile
        else if (program_path.find("test") != std::string::npos) adaptive_timeout_ms = 60000; // 1 min for test
        else if (program_path.find("network") != std::string::npos) adaptive_timeout_ms = 45000; // 45s for network
    } else {
        adaptive_timeout_ms = static_cast<DWORD>(timeout.count() * 1000);
    }
    
    DWORD waitResult = WaitForSingleObject(pi.hProcess, adaptive_timeout_ms);
    bool timedOut = (waitResult == WAIT_TIMEOUT);

    DWORD exitCode = 0;
    GetExitCodeProcess(pi.hProcess, &exitCode);

    // Read output
    std::string output;
    char buffer[4096];
    DWORD bytesRead;
    while (ReadFile(hReadPipe, buffer, sizeof(buffer) - 1, &bytesRead, NULL) && bytesRead > 0) {
        buffer[bytesRead] = '\0';
        output += buffer;
    }

    // Cleanup
    CloseHandle(hReadPipe);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (timedOut) {
        return {false, -1, output, "", duration, "Process timed out"};
    }

    return {true, static_cast<int>(exitCode), output, "", duration, ""};
}
#endif

ProgramExecutionResult SystemIntegrationManager::execute_program_macos(const std::string& program_path,
                                                                     const std::vector<std::string>& arguments,
                                                                     std::chrono::seconds timeout) {
    // macOS implementation
    return {false, -1, "", "", std::chrono::milliseconds(0), "Not implemented for macOS"};
}

void SystemIntegrationManager::initialize_safety_rules() {
    // Define safe paths
    safe_paths_ = {
        home_directory_,
        home_directory_ + "/Documents",
        home_directory_ + "/Downloads",
        home_directory_ + "/Desktop",
        home_directory_ + "/.config"
    };

    // Define dangerous commands
    dangerous_commands_ = {
        "rm", "rmdir", "del", "format", "fdisk", "mkfs",
        "dd", "shred", "wipe", "sudo", "su", "passwd",
        "chmod", "chown", "mount", "umount", "kill", "pkill"
    };
}

bool SystemIntegrationManager::is_command_safe(const std::string& command) const {
    for (const auto& dangerous : dangerous_commands_) {
        if (command.find(dangerous) != std::string::npos) {
            return false;
        }
    }

    return !contains_dangerous_patterns(command);
}

bool SystemIntegrationManager::contains_dangerous_patterns(const std::string& input) const {
    // Check for dangerous patterns
    std::vector<std::string> patterns = {
        "..", "/etc", "/root", "/sys", "/proc", "/dev",
        ">", ">>", "|", ";", "&&", "||", "`", "$("
    };

    for (const auto& pattern : patterns) {
        if (input.find(pattern) != std::string::npos) {
            return true;
        }
    }

    return false;
}

std::string SystemIntegrationManager::sanitize_path(const std::string& path) const {
    std::string sanitized = path;

    // Remove dangerous path components
    std::vector<std::string> dangerous = {"..", "//", "~"};

    for (const auto& danger : dangerous) {
        size_t pos = 0;
        while ((pos = sanitized.find(danger, pos)) != std::string::npos) {
            sanitized.erase(pos, danger.length());
        }
    }

    return sanitized;
}

SystemType SystemIntegrationManager::detect_system_type() {
#ifdef __linux__
    return SystemType::LINUX;
#elif _WIN32
    return SystemType::WINDOWS;
#elif __APPLE__
    return SystemType::MACOS;
#else
    return SystemType::UNKNOWN;
#endif
}

std::string SystemIntegrationManager::get_home_directory() {
#ifdef _WIN32
    const char* home = std::getenv("USERPROFILE");
    if (home) {
        return home;
    }
    // Fallback to APPDATA or current dir
    const char* appdata = std::getenv("APPDATA");
    if (appdata) {
        return appdata;
    }
    return "."; // Current directory fallback
#else
    const char* home = std::getenv("HOME");
    if (home) {
        return home;
    }

    // Fallback for systems without HOME
    struct passwd* pw = getpwuid(getuid());
    if (pw) {
        return pw->pw_dir;
    }

    return "/tmp"; // Ultimate fallback
#endif
}

std::string SystemIntegrationManager::get_system_root() {
#ifdef _WIN32
    return "C:\\"; // Windows system root
#else
    return "/"; // Unix-like systems
#endif
}

std::string SystemIntegrationManager::build_command_string(const std::string& program_path,
                                                         const std::vector<std::string>& arguments) {
    std::string command = program_path;

    for (const auto& arg : arguments) {
        command += " \"" + arg + "\""; // Quote arguments for safety
    }

    return command;
}

ProgramExecutionResult SystemIntegrationManager::parse_execution_output(const std::string& output,
                                                                       int exit_code,
                                                                       std::chrono::milliseconds duration) {
    ProgramExecutionResult result;
    result.success = (exit_code == 0);
    result.exit_code = exit_code;
    result.stdout_output = output;
    result.execution_time = duration;

    // In a real implementation, we'd separate stdout and stderr
    result.stderr_output = "";

    if (!result.success) {
        result.error_message = "Command failed with exit code " + std::to_string(exit_code);
    }

    return result;
}