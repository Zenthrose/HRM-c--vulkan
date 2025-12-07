#pragma once

#ifdef SPDLOG_AVAILABLE
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#endif

#include <memory>
#include <string>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <mutex>
#include <chrono>

namespace fs = std::filesystem;

class Logger {
public:
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

#ifdef SPDLOG_AVAILABLE
    std::shared_ptr<spdlog::logger> getLogger() {
        return logger_;
    }

    void setLogLevel(spdlog::level::level_enum level) {
        logger_->set_level(level);
    }
#endif

    void debug(const std::string& message) {
#ifdef SPDLOG_AVAILABLE
        logger_->debug(message);
#else
        log("DEBUG", message);
#endif
    }

    void info(const std::string& message) {
#ifdef SPDLOG_AVAILABLE
        logger_->info(message);
#else
        log("INFO", message);
#endif
    }

    void warning(const std::string& message) {
#ifdef SPDLOG_AVAILABLE
        logger_->warn(message);
#else
        log("WARNING", message);
#endif
    }

    void error(const std::string& message) {
#ifdef SPDLOG_AVAILABLE
        logger_->error(message);
#else
        log("ERROR", message);
#endif
    }

private:
    Logger() {
#ifdef SPDLOG_AVAILABLE
        // Create console sink
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

        std::vector<spdlog::sink_ptr> sinks = {console_sink};

        // Add file sink if HRM_LOG_DIR is set
        const char* log_dir = std::getenv("HRM_LOG_DIR");
        if (log_dir) {
            fs::path log_path = fs::path(log_dir) / "hrm_system.log";
            auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_path.string(), true);
            file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");
            sinks.push_back(file_sink);
        }

        // Create logger
        logger_ = std::make_shared<spdlog::logger>("hrm_logger", sinks.begin(), sinks.end());

        // Set log level from environment variable
        const char* env_level = std::getenv("HRM_LOG_LEVEL");
        spdlog::level::level_enum level = spdlog::level::info; // Default
        if (env_level) {
            std::string level_str = env_level;
            if (level_str == "debug") level = spdlog::level::debug;
            else if (level_str == "info") level = spdlog::level::info;
            else if (level_str == "warning") level = spdlog::level::warn;
            else if (level_str == "error") level = spdlog::level::err;
            else if (level_str == "none") level = spdlog::level::off;
        }
        logger_->set_level(level);

        // Set as default logger
        spdlog::set_default_logger(logger_);
#endif
    }

#ifdef SPDLOG_AVAILABLE
    ~Logger() {
        spdlog::shutdown();
    }
#endif

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

#ifndef SPDLOG_AVAILABLE
    // Fallback logging implementation when spdlog is not available
    void log(const std::string& level, const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Get current timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = *std::localtime(&time_t);

        // Format timestamp
        char timestamp[20];
        std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", &tm);

        std::string log_message = std::string("[") + timestamp + "] [" + level + "] " + message;

        // Console output
        std::cout << log_message << std::endl;

        // File output if HRM_LOG_DIR is set
        const char* log_dir = std::getenv("HRM_LOG_DIR");
        if (log_dir) {
            fs::path log_path = fs::path(log_dir) / "hrm_system.log";
            std::ofstream log_file(log_path, std::ios::app);
            if (log_file.is_open()) {
                log_file << log_message << std::endl;
            }
        }
    }

    std::mutex mutex_;
#endif

#ifdef SPDLOG_AVAILABLE
    std::shared_ptr<spdlog::logger> logger_;
#endif
};