#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <memory>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

class Logger {
public:
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    std::shared_ptr<spdlog::logger> getLogger() {
        return logger_;
    }

    void setLogLevel(spdlog::level::level_enum level) {
        logger_->set_level(level);
    }

    void debug(const std::string& message) { logger_->debug(message); }
    void info(const std::string& message) { logger_->info(message); }
    void warning(const std::string& message) { logger_->warn(message); }
    void error(const std::string& message) { logger_->error(message); }

private:
    Logger() {
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
    }

    ~Logger() {
        spdlog::shutdown();
    }

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::shared_ptr<spdlog::logger> logger_;
};