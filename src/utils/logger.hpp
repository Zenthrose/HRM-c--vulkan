#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <memory>
#include <mutex>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>

#ifdef ERROR
#undef ERROR
#endif

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    NONE
};

class Logger {
public:
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    void setLogLevel(LogLevel level) {
        std::lock_guard<std::mutex> lock(mutex_);
        log_level_ = level;
    }

    void setLogFile(const std::string& filename) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (log_file_.is_open()) {
            log_file_.close();
        }
        log_file_.open(filename, std::ios::app);
        if (!log_file_.is_open()) {
            std::cerr << "Failed to open log file: " << filename << std::endl;
        }
    }

    void log(LogLevel level, const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (level < log_level_) {
            return;
        }

        std::string level_str;
        switch (level) {
            case LogLevel::DEBUG: level_str = "DEBUG"; break;
            case LogLevel::INFO: level_str = "INFO"; break;
            case LogLevel::WARNING: level_str = "WARNING"; break;
            case LogLevel::ERROR: level_str = "ERROR"; break;
            default: level_str = "UNKNOWN"; break;
        }

        std::string log_message = "[" + level_str + "] " + message;

        // Console output
        std::cout << log_message << std::endl;

        // File output
        if (log_file_.is_open()) {
            log_file_ << log_message << std::endl;
            log_file_.flush();
        }
    }

    void debug(const std::string& message) { log(LogLevel::DEBUG, message); }
    void info(const std::string& message) { log(LogLevel::INFO, message); }
    void warning(const std::string& message) { log(LogLevel::WARNING, message); }
    void error(const std::string& message) { log(LogLevel::ERROR, message); }

private:
    Logger() {
        // Initialize with environment variable
        const char* env_level = std::getenv("HRM_LOG_LEVEL");
        if (env_level) {
            if (std::string(env_level) == "debug") log_level_ = LogLevel::DEBUG;
            else if (std::string(env_level) == "info") log_level_ = LogLevel::INFO;
            else if (std::string(env_level) == "warning") log_level_ = LogLevel::WARNING;
            else if (std::string(env_level) == "error") log_level_ = LogLevel::ERROR;
            else if (std::string(env_level) == "none") log_level_ = LogLevel::NONE;
        } else {
            log_level_ = LogLevel::INFO; // Default
        }

        // Initialize log file if HRM_LOG_DIR is set
        const char* log_dir = std::getenv("HRM_LOG_DIR");
        if (log_dir) {
            fs::path log_path = fs::path(log_dir) / "hrm_system.log";
            setLogFile(log_path.string());
        }
    }

    ~Logger() {
        if (log_file_.is_open()) {
            log_file_.close();
        }
    }

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    LogLevel log_level_;
    std::ofstream log_file_;
    std::mutex mutex_;
};