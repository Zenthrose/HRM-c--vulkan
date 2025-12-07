#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <filesystem>

namespace fs = std::filesystem;

class ConfigManager {
public:
    ConfigManager(std::string config_dir = "") {
        if (config_dir.empty()) {
            if (const char* env = std::getenv("HRM_CONFIG_DIR")) {
                config_dir = env;
            } else {
                std::string home;
                if (const char* h = std::getenv("HOME")) home = h;
                else if (const char* h = std::getenv("USERPROFILE")) home = h;
                if (!home.empty()) {
                    config_dir = home + "/.hrm/config";
                } else {
                    config_dir = "./config";
                }
            }
        }
        config_dir_ = config_dir;
        fs::create_directories(config_dir_);
    }
    ~ConfigManager() {}

    // Configuration loading/saving
    bool load_config(const std::string& filename = "hrm_config.txt");
    bool save_config(const std::string& filename = "hrm_config.txt");

    // Value access
    std::string get_string(const std::string& key, const std::string& default_value = "") const;
    int get_int(const std::string& key, int default_value = 0) const;
    double get_double(const std::string& key, double default_value = 0.0) const;
    bool get_bool(const std::string& key, bool default_value = false) const;

    // Value setting
    void set_string(const std::string& key, const std::string& value);
    void set_int(const std::string& key, int value);
    void set_double(const std::string& key, double value);
    void set_bool(const std::string& key, bool value);

    bool has_key(const std::string& key) const;
    std::vector<std::string> get_keys() const;
    void clear();

    // Cloud provider configuration
    struct CloudProviderConfig {
        std::string provider_name;
        std::string api_key;
        std::string client_id;
        std::string client_secret;
        std::string refresh_token;
        std::string access_token;
        std::string root_folder_id;
        bool enabled;
    };

    CloudProviderConfig get_cloud_config(const std::string& provider);
    void set_cloud_config(const std::string& provider, const CloudProviderConfig& config);

    // Directory access
    std::string getConfigDir() const { return config_dir_; }

private:
    std::string config_dir_;
    std::unordered_map<std::string, std::string> config_data_;
};