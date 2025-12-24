#include "config_manager.hpp"
#include <fstream>
#include <sstream>

bool ConfigManager::load_config(const std::string& filename) {
    std::ifstream file(config_dir_ + "/" + filename);
    if (!file) return false;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            config_data_[key] = value;
        }
    }
    return true;
}

bool ConfigManager::save_config(const std::string& filename) {
    std::ofstream file(config_dir_ + "/" + filename);
    if (!file) return false;
    for (const auto& p : config_data_) {
        file << p.first << "=" << p.second << std::endl;
    }
    return true;
}

std::string ConfigManager::get_string(const std::string& key, const std::string& default_value) const {
    auto it = config_data_.find(key);
    return it != config_data_.end() ? it->second : default_value;
}

int ConfigManager::get_int(const std::string& key, int default_value) const {
    std::string val = get_string(key);
    if (val.empty()) return default_value;
    return std::stoi(val);
}

double ConfigManager::get_double(const std::string& key, double default_value) const {
    std::string val = get_string(key);
    if (val.empty()) return default_value;
    return std::stod(val);
}

bool ConfigManager::get_bool(const std::string& key, bool default_value) const {
    std::string val = get_string(key);
    if (val.empty()) return default_value;
    return val == "true" || val == "1";
}

void ConfigManager::set_string(const std::string& key, const std::string& value) {
    config_data_[key] = value;
}

void ConfigManager::set_int(const std::string& key, int value) {
    set_string(key, std::to_string(value));
}

void ConfigManager::set_double(const std::string& key, double value) {
    set_string(key, std::to_string(value));
}

void ConfigManager::set_bool(const std::string& key, bool value) {
    set_string(key, value ? "true" : "false");
}

bool ConfigManager::has_key(const std::string& key) const {
    return config_data_.count(key) > 0;
}

std::vector<std::string> ConfigManager::get_keys() const {
    std::vector<std::string> keys;
    for (const auto& p : config_data_) {
        keys.push_back(p.first);
    }
    return keys;
}

void ConfigManager::clear() {
    config_data_.clear();
}

// Stub implementations for cloud config
ConfigManager::CloudProviderConfig ConfigManager::get_cloud_config(const std::string& provider) {
    CloudProviderConfig config;
    config.provider_name = get_string(provider + ".name");
    config.api_key = get_string(provider + ".api_key");
    config.client_id = get_string(provider + ".client_id");
    config.client_secret = get_string(provider + ".client_secret");
    config.refresh_token = get_string(provider + ".refresh_token");
    config.access_token = get_string(provider + ".access_token");
    config.root_folder_id = get_string(provider + ".root_folder_id");
    config.enabled = get_bool(provider + ".enabled", false);
    return config;
}

void ConfigManager::set_cloud_config(const std::string& provider, const CloudProviderConfig& config) {
    set_string(provider + ".name", config.provider_name);
    set_string(provider + ".api_key", config.api_key);
    set_string(provider + ".client_id", config.client_id);
    set_string(provider + ".client_secret", config.client_secret);
    set_string(provider + ".refresh_token", config.refresh_token);
    set_string(provider + ".access_token", config.access_token);
    set_string(provider + ".root_folder_id", config.root_folder_id);
    set_bool(provider + ".enabled", config.enabled);
}