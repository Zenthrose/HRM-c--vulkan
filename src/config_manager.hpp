#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <filesystem>

namespace fs = std::filesystem;

class ConfigManager {
public:
    ConfigManager(const std::string& config_dir = "./config");
    ~ConfigManager();

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

    // Section management
    void set_section(const std::string& section);
    std::string get_current_section() const;

    // Utility
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

private:
    std::string config_dir_;
    std::unordered_map<std::string, std::string> config_data_;
    std::string current_section_;

    std::string make_key(const std::string& key) const;
    void parse_line(const std::string& line);
    std::string escape_value(const std::string& value) const;
    std::string unescape_value(const std::string& value) const;
};