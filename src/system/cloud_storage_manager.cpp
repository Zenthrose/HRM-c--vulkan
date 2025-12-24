#include "cloud_storage_manager.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <iomanip>
#include <stdexcept>
#include <memory>
#include "resource_monitor.hpp"

CloudStorageProvider::CloudStorageProvider(const CloudStorageConfig& config)
    : config_(config), last_auth_time_(std::chrono::system_clock::now()) {
}

std::string CloudStorageProvider::get_provider_name() const {
    switch (config_.provider) {
        case CloudProvider::GOOGLE_DRIVE: return "Google Drive";
        case CloudProvider::DROPBOX: return "Dropbox";
        case CloudProvider::ONEDRIVE: return "OneDrive";
        case CloudProvider::MEGA: return "Mega";
        case CloudProvider::LOCAL_STORAGE: return "Local Storage";
        default: return "Unknown";
    }
}

std::string CloudStorageProvider::generate_unique_filename(const std::string& base_name) {
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();

    std::stringstream ss;
    ss << base_name << "_" << timestamp << "_" << rand() % 1000;
    return ss.str();
}

bool CloudStorageProvider::validate_config() const {
    if (config_.provider == CloudProvider::LOCAL_STORAGE) {
        return true; // Local storage doesn't need API keys
    }

    // For cloud providers, check required credentials
    if (config_.api_key.empty() && config_.client_id.empty()) {
        return false;
    }

    return true;
}

std::string CloudStorageProvider::url_encode(const std::string& str) const {
    std::stringstream ss;
    for (char c : str) {
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            ss << c;
        } else {
            ss << '%' << std::hex << std::setw(2) << std::setfill('0')
               << static_cast<int>(static_cast<unsigned char>(c));
        }
    }
    return ss.str();
}

std::string CloudStorageProvider::base64_encode(const std::vector<uint8_t>& data) const {
    // Simple base64 encoding (placeholder - would use proper library)
    static const char* base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string result;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    for (size_t idx = 0; idx < data.size(); ++idx) {
        char_array_3[i++] = data[idx];
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; i < 4; i++) {
                result += base64_chars[char_array_4[i]];
            }
            i = 0;
        }
    }

    // Handle remaining bytes
    if (i) {
        for (j = i; j < 3; j++) {
            char_array_3[j] = '\0';
        }

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (j = 0; j < i + 1; j++) {
            result += base64_chars[char_array_4[j]];
        }

        while (i++ < 3) {
            result += '=';
        }
    }

    return result;
}

// Google Drive Implementation (Framework - would need actual API integration)
GoogleDriveProvider::GoogleDriveProvider(const CloudStorageConfig& config)
    : CloudStorageProvider(config) {
    api_base_url_ = "https://www.googleapis.com/drive/v3";
    upload_url_ = "https://www.googleapis.com/upload/drive/v3";
}

bool GoogleDriveProvider::authenticate() {
    // Local file-based fallback authentication
    std::cout << "Google Drive authentication (local file fallback)" << std::endl;
    
    // Ensure local storage directory exists
    std::string local_dir = config_.root_folder_id.empty() ? "/tmp/hrm_cloud_storage" : config_.root_folder_id;
    try {
        std::filesystem::create_directories(local_dir);
        last_auth_time_ = std::chrono::system_clock::now();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create local storage directory: " << e.what() << std::endl;
        return false;
    }
}

bool GoogleDriveProvider::refresh_token() {
    // Local storage doesn't need token refresh
    return true;
}

bool GoogleDriveProvider::is_authenticated() const {
    // Local storage always authenticated
    return true;
}

UploadResult GoogleDriveProvider::upload_file(const std::string& local_path,
                                            const std::string& remote_name,
                                            const std::unordered_map<std::string, std::string>& metadata) {
    UploadResult result;
    result.success = false;
    
    try {
        if (!std::filesystem::exists(local_path)) {
            result.error_message = "Local file does not exist: " + local_path;
            return result;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Copy file to local storage with unique ID
        std::string storage_dir = config_.root_folder_id.empty() ? "/tmp/hrm_cloud_storage" : config_.root_folder_id;
        std::string file_id = generate_unique_filename(remote_name);
        std::string storage_path = storage_dir + "/" + file_id;
        
        std::filesystem::copy_file(local_path, storage_path);
        
        auto file_size = std::filesystem::file_size(local_path);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        result.success = true;
        result.file_id = file_id;
        result.uploaded_size = file_size;
        result.upload_time = duration;
        
        std::cout << "Uploaded to local storage: " << file_id << std::endl;
    } catch (const std::exception& e) {
        result.error_message = std::string("Upload failed: ") + e.what();
    }
    
    return result;
}

UploadResult GoogleDriveProvider::upload_data(const std::vector<uint8_t>& data,
                                            const std::string& remote_name,
                                            const std::unordered_map<std::string, std::string>& metadata) {
    UploadResult result;
    result.success = false;
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Write data to local storage
        std::string storage_dir = config_.root_folder_id.empty() ? "/tmp/hrm_cloud_storage" : config_.root_folder_id;
        std::filesystem::create_directories(storage_dir);
        std::string file_id = generate_unique_filename(remote_name);
        std::string storage_path = storage_dir + "/" + file_id;
        
        std::ofstream file(storage_path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(data.data()), data.size());
        file.close();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        result.success = true;
        result.file_id = file_id;
        result.uploaded_size = data.size();
        result.upload_time = duration;
        
        std::cout << "Uploaded data to local storage: " << file_id << std::endl;
    } catch (const std::exception& e) {
        result.error_message = std::string("Upload failed: ") + e.what();
    }
    
    return result;
}

DownloadResult GoogleDriveProvider::download_file(const std::string& file_id) {
    DownloadResult result;
    result.success = false;
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::string storage_dir = config_.root_folder_id.empty() ? "/tmp/hrm_cloud_storage" : config_.root_folder_id;
        std::string storage_path = storage_dir + "/" + file_id;
        
        if (!std::filesystem::exists(storage_path)) {
            result.error_message = "File not found in local storage: " + file_id;
            return result;
        }
        
        // Read file from local storage
        std::ifstream file(storage_path, std::ios::binary);
        std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        
        result.data = data;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        result.success = true;
        result.downloaded_size = data.size();
        result.download_time = duration;
        
        std::cout << "Downloaded from local storage: " << file_id << std::endl;
    } catch (const std::exception& e) {
        result.error_message = std::string("Download failed: ") + e.what();
    }
    
    return result;
}

bool GoogleDriveProvider::delete_file(const std::string& file_id) {
    try {
        std::string storage_dir = config_.root_folder_id.empty() ? "/tmp/hrm_cloud_storage" : config_.root_folder_id;
        std::string storage_path = storage_dir + "/" + file_id;
        return std::filesystem::remove(storage_path) > 0;
    } catch (const std::exception& e) {
        std::cerr << "Failed to delete file: " << e.what() << std::endl;
        return false;
    }
}

std::vector<CloudFile> GoogleDriveProvider::list_files(const std::string& folder_id) {
    std::vector<CloudFile> files;
    
    try {
        std::string storage_dir = config_.root_folder_id.empty() ? "/tmp/hrm_cloud_storage" : config_.root_folder_id;
        
        for (const auto& entry : std::filesystem::directory_iterator(storage_dir)) {
            if (entry.is_regular_file()) {
                CloudFile file;
                file.file_id = entry.path().filename().string();
                file.name = entry.path().filename().string();
                file.path = entry.path().string();
                file.size_bytes = std::filesystem::file_size(entry);
                file.created_time = std::chrono::system_clock::now();
                file.modified_time = file.created_time;
                files.push_back(file);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to list files: " << e.what() << std::endl;
    }
    
    return files;
}

std::string GoogleDriveProvider::create_folder(const std::string& name, const std::string& parent_id) {
    try {
        std::string storage_dir = config_.root_folder_id.empty() ? "/tmp/hrm_cloud_storage" : config_.root_folder_id;
        std::string folder_path = storage_dir + "/" + name;
        std::filesystem::create_directories(folder_path);
        return folder_path;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create folder: " << e.what() << std::endl;
        return "";
    }
}

std::string GoogleDriveProvider::get_folder_id(const std::string& folder_name) {
    return config_.root_folder_id.empty() ? "/tmp/hrm_cloud_storage" : config_.root_folder_id;
}

std::string GoogleDriveProvider::make_api_request(const std::string& url, const std::string& method,
                                                const std::string& data, const std::string& content_type) {
    // Local storage doesn't use API requests
    return "{\"success\":true}";
}

// Dropbox Implementation (Framework)
DropboxProvider::DropboxProvider(const CloudStorageConfig& config)
    : CloudStorageProvider(config) {
    api_base_url_ = "https://api.dropboxapi.com/2";
    content_url_ = "https://content.dropboxapi.com/2";
}

bool DropboxProvider::authenticate() {
    // Local file-based fallback for Dropbox
    std::cout << "Dropbox authentication (local file fallback)" << std::endl;
    
    try {
        std::string local_dir = config_.root_folder_id.empty() ? "/tmp/hrm_dropbox_storage" : config_.root_folder_id;
        std::filesystem::create_directories(local_dir);
        last_auth_time_ = std::chrono::system_clock::now();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create local storage: " << e.what() << std::endl;
        return false;
    }
}

bool DropboxProvider::refresh_token() { return true; }
bool DropboxProvider::is_authenticated() const { return true; }

UploadResult DropboxProvider::upload_file(const std::string& local_path,
                                        const std::string& remote_name,
                                        const std::unordered_map<std::string, std::string>& metadata) {
    UploadResult result;
    result.success = false;

    if (!is_authenticated()) {
        result.error_message = "Not authenticated with Dropbox";
        return result;
    }

    try {
        auto start_time = std::chrono::high_resolution_clock::now();

        if (!std::filesystem::exists(local_path)) {
            result.error_message = "Local file does not exist: " + local_path;
            return result;
        }

        // Store file to local storage with unique ID
        std::string storage_dir = config_.root_folder_id.empty() ? "/tmp/hrm_dropbox_storage" : config_.root_folder_id;
        std::string file_id = generate_unique_filename(remote_name);
        std::string storage_path = storage_dir + "/" + file_id;
        
        std::filesystem::copy_file(local_path, storage_path);

        auto file_size = std::filesystem::file_size(local_path);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        result.success = true;
        result.file_id = file_id;
        result.uploaded_size = file_size;
        result.upload_time = duration;

    } catch (const std::exception& e) {
        result.error_message = std::string("Upload failed: ") + e.what();
    }

    return result;
}

UploadResult DropboxProvider::upload_data(const std::vector<uint8_t>& data,
                                        const std::string& remote_name,
                                        const std::unordered_map<std::string, std::string>& metadata) {
    UploadResult result;
    result.success = false;

    if (!is_authenticated()) {
        result.error_message = "Not authenticated with Dropbox";
        return result;
    }

    try {
        auto start_time = std::chrono::high_resolution_clock::now();

        std::string storage_dir = config_.root_folder_id.empty() ? "/tmp/hrm_dropbox_storage" : config_.root_folder_id;
        std::filesystem::create_directories(storage_dir);
        std::string file_id = generate_unique_filename(remote_name);
        std::string storage_path = storage_dir + "/" + file_id;
        
        std::ofstream file(storage_path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(data.data()), data.size());
        file.close();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        result.success = true;
        result.file_id = file_id;
        result.uploaded_size = data.size();
        result.upload_time = duration;

    } catch (const std::exception& e) {
        result.error_message = std::string("Upload failed: ") + e.what();
    }

    return result;
}

DownloadResult DropboxProvider::download_file(const std::string& file_id) {
    DownloadResult result;
    result.success = false;

    if (!is_authenticated()) {
        result.error_message = "Not authenticated with Dropbox";
        return result;
    }

    try {
        auto start_time = std::chrono::high_resolution_clock::now();

        std::string storage_dir = config_.root_folder_id.empty() ? "/tmp/hrm_dropbox_storage" : config_.root_folder_id;
        std::string storage_path = storage_dir + "/" + file_id;
        
        if (!std::filesystem::exists(storage_path)) {
            result.error_message = "File not found in local storage: " + file_id;
            return result;
        }
        
        std::ifstream file(storage_path, std::ios::binary);
        std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        
        result.data = data;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        result.success = true;
        result.downloaded_size = result.data.size();
        result.download_time = duration;

    } catch (const std::exception& e) {
        result.error_message = std::string("Download failed: ") + e.what();
    }

    return result;
}

bool DropboxProvider::delete_file(const std::string& file_id) {
    try {
        std::string storage_dir = config_.root_folder_id.empty() ? "/tmp/hrm_dropbox_storage" : config_.root_folder_id;
        std::string storage_path = storage_dir + "/" + file_id;
        return std::filesystem::remove(storage_path) > 0;
    } catch (const std::exception& e) {
        return false;
    }
}
std::vector<CloudFile> DropboxProvider::list_files(const std::string& folder_id) {
    std::vector<CloudFile> files;

    if (!is_authenticated()) {
        return files;
    }

    // Simulate API call to list files
    std::string api_response = make_api_request("/files/list_folder", "POST",
                                               "{\"path\":\"" + folder_id + "\"}",
                                               "application/json");

    if (api_response.find("entries") != std::string::npos) {
        // Parse simulated response (in real implementation, would parse JSON)
        CloudFile file1;
        file1.file_id = "dropbox_file_1";
        file1.name = "uploaded_file.txt";
        file1.path = "/uploaded_file.txt";
        file1.size_bytes = 1024;
        file1.created_time = std::chrono::system_clock::now();
        file1.modified_time = file1.created_time;

        CloudFile file2;
        file2.file_id = "dropbox_file_2";
        file2.name = "memory_compaction.mem";
        file2.path = "/memory_compaction.mem";
        file2.size_bytes = 2048;
        file2.created_time = std::chrono::system_clock::now();
        file2.modified_time = file2.created_time;

        files.push_back(file1);
        files.push_back(file2);
    }

    return files;
}
std::string DropboxProvider::create_folder(const std::string& name, const std::string& parent_id) { return ""; }

std::string DropboxProvider::make_api_request(const std::string& endpoint, const std::string& method,
                                             const std::string& data, const std::string& content_type) {
    // Simulate Dropbox API response
    // In a full implementation, this would make actual HTTPS requests

    if (endpoint == "/files/upload") {
        // Simulate successful upload response
        return "{\"name\":\"uploaded_file\",\"path_lower\":\"/uploaded_file\",\"id\":\"id:abc123\",\"size\":1024}";
    } else if (endpoint == "/files/download") {
        // Simulate successful download response
        return "{\"name\":\"downloaded_file\",\"path_lower\":\"/downloaded_file\",\"id\":\"id:def456\",\"size\":2048}";
    } else if (endpoint == "/files/list_folder") {
        // Simulate file listing
        return "{\"entries\":[{\"name\":\"file1.txt\",\"id\":\"id:123\",\"size\":100},{\"name\":\"file2.txt\",\"id\":\"id:456\",\"size\":200}]}";
    }

    // Default success response
    return "{\"success\":true}";
}

// OneDrive Implementation (Framework)
OneDriveProvider::OneDriveProvider(const CloudStorageConfig& config)
    : CloudStorageProvider(config) {
    api_base_url_ = "https://graph.microsoft.com/v1.0/me/drive";
}

bool OneDriveProvider::authenticate() {
    // Local file-based fallback for OneDrive
    std::cout << "OneDrive authentication (local file fallback)" << std::endl;
    
    try {
        std::string local_dir = config_.root_folder_id.empty() ? "/tmp/hrm_onedrive_storage" : config_.root_folder_id;
        std::filesystem::create_directories(local_dir);
        last_auth_time_ = std::chrono::system_clock::now();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create local storage: " << e.what() << std::endl;
        return false;
    }
}

bool OneDriveProvider::refresh_token() { return true; }
bool OneDriveProvider::is_authenticated() const { return true; }

UploadResult OneDriveProvider::upload_file(const std::string& local_path,
                                         const std::string& remote_name,
                                         const std::unordered_map<std::string, std::string>& metadata) {
    UploadResult result;
    result.success = false;
    
    try {
        if (!std::filesystem::exists(local_path)) {
            result.error_message = "Local file does not exist: " + local_path;
            return result;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::string storage_dir = config_.root_folder_id.empty() ? "/tmp/hrm_onedrive_storage" : config_.root_folder_id;
        std::string file_id = generate_unique_filename(remote_name);
        std::string storage_path = storage_dir + "/" + file_id;
        
        std::filesystem::copy_file(local_path, storage_path);
        
        auto file_size = std::filesystem::file_size(local_path);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        result.success = true;
        result.file_id = file_id;
        result.uploaded_size = file_size;
        result.upload_time = duration;
    } catch (const std::exception& e) {
        result.error_message = std::string("Upload failed: ") + e.what();
    }
    
    return result;
}

UploadResult OneDriveProvider::upload_data(const std::vector<uint8_t>& data,
                                         const std::string& remote_name,
                                         const std::unordered_map<std::string, std::string>& metadata) {
    UploadResult result;
    result.success = false;
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::string storage_dir = config_.root_folder_id.empty() ? "/tmp/hrm_onedrive_storage" : config_.root_folder_id;
        std::filesystem::create_directories(storage_dir);
        std::string file_id = generate_unique_filename(remote_name);
        std::string storage_path = storage_dir + "/" + file_id;
        
        std::ofstream file(storage_path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(data.data()), data.size());
        file.close();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        result.success = true;
        result.file_id = file_id;
        result.uploaded_size = data.size();
        result.upload_time = duration;
    } catch (const std::exception& e) {
        result.error_message = std::string("Upload failed: ") + e.what();
    }
    
    return result;
}

DownloadResult OneDriveProvider::download_file(const std::string& file_id) {
    DownloadResult result;
    result.success = false;
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::string storage_dir = config_.root_folder_id.empty() ? "/tmp/hrm_onedrive_storage" : config_.root_folder_id;
        std::string storage_path = storage_dir + "/" + file_id;
        
        if (!std::filesystem::exists(storage_path)) {
            result.error_message = "File not found in local storage: " + file_id;
            return result;
        }
        
        std::ifstream file(storage_path, std::ios::binary);
        std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        
        result.data = data;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        result.success = true;
        result.downloaded_size = data.size();
        result.download_time = duration;
    } catch (const std::exception& e) {
        result.error_message = std::string("Download failed: ") + e.what();
    }
    
    return result;
}

bool OneDriveProvider::delete_file(const std::string& file_id) { 
    try {
        std::string storage_dir = config_.root_folder_id.empty() ? "/tmp/hrm_onedrive_storage" : config_.root_folder_id;
        std::string storage_path = storage_dir + "/" + file_id;
        return std::filesystem::remove(storage_path) > 0;
    } catch (const std::exception& e) {
        return false;
    }
}
std::vector<CloudFile> OneDriveProvider::list_files(const std::string& folder_id) { return {}; }
std::string OneDriveProvider::create_folder(const std::string& name, const std::string& parent_id) { return ""; }

std::string OneDriveProvider::make_api_request(const std::string& endpoint, const std::string& method,
                                             const std::string& data, const std::string& content_type) {
    return "{}";
}

// Mega Implementation (Framework)
MegaProvider::MegaProvider(const CloudStorageConfig& config)
    : CloudStorageProvider(config) {
}

bool MegaProvider::authenticate() { return true; }
bool MegaProvider::refresh_token() { return true; }
bool MegaProvider::is_authenticated() const { return true; }

UploadResult MegaProvider::upload_file(const std::string& local_path,
                                     const std::string& remote_name,
                                     const std::unordered_map<std::string, std::string>& metadata) {
    UploadResult result;
    result.success = false;
    
    try {
        if (!std::filesystem::exists(local_path)) {
            result.error_message = "Local file does not exist: " + local_path;
            return result;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Mega fallback: use local file-based storage
        std::string storage_dir = config_.root_folder_id.empty() ? "/tmp/hrm_mega_storage" : config_.root_folder_id;
        std::filesystem::create_directories(storage_dir);
        std::string file_id = generate_unique_filename(remote_name);
        std::string storage_path = storage_dir + "/" + file_id;
        
        std::filesystem::copy_file(local_path, storage_path);
        
        auto file_size = std::filesystem::file_size(local_path);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        result.success = true;
        result.file_id = file_id;
        result.uploaded_size = file_size;
        result.upload_time = duration;
        
        std::cout << "Uploaded to Mega (local fallback): " << file_id << std::endl;
    } catch (const std::exception& e) {
        result.error_message = std::string("Upload failed: ") + e.what();
    }
    
    return result;
}

UploadResult MegaProvider::upload_data(const std::vector<uint8_t>& data,
                                     const std::string& remote_name,
                                     const std::unordered_map<std::string, std::string>& metadata) {
    UploadResult result;
    result.success = false;
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Mega fallback: use local file-based storage
        std::string storage_dir = config_.root_folder_id.empty() ? "/tmp/hrm_mega_storage" : config_.root_folder_id;
        std::filesystem::create_directories(storage_dir);
        std::string file_id = generate_unique_filename(remote_name);
        std::string storage_path = storage_dir + "/" + file_id;
        
        std::ofstream file(storage_path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(data.data()), data.size());
        file.close();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        result.success = true;
        result.file_id = file_id;
        result.uploaded_size = data.size();
        result.upload_time = duration;
        
        std::cout << "Uploaded data to Mega (local fallback): " << file_id << std::endl;
    } catch (const std::exception& e) {
        result.error_message = std::string("Upload failed: ") + e.what();
    }
    
    return result;
}

DownloadResult MegaProvider::download_file(const std::string& file_id) {
    DownloadResult result;
    result.success = false;
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Mega fallback: retrieve from local storage
        std::string storage_dir = config_.root_folder_id.empty() ? "/tmp/hrm_mega_storage" : config_.root_folder_id;
        std::string storage_path = storage_dir + "/" + file_id;
        
        if (!std::filesystem::exists(storage_path)) {
            result.error_message = "File not found in Mega storage: " + file_id;
            return result;
        }
        
        std::ifstream file(storage_path, std::ios::binary);
        std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        
        result.data = data;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        result.success = true;
        result.downloaded_size = data.size();
        result.download_time = duration;
        
        std::cout << "Downloaded from Mega (local fallback): " << file_id << std::endl;
    } catch (const std::exception& e) {
        result.error_message = std::string("Download failed: ") + e.what();
    }
    
    return result;
}

bool MegaProvider::delete_file(const std::string& file_id) { 
    try {
        std::string storage_dir = config_.root_folder_id.empty() ? "/tmp/hrm_mega_storage" : config_.root_folder_id;
        std::string storage_path = storage_dir + "/" + file_id;
        return std::filesystem::remove(storage_path) > 0;
    } catch (const std::exception& e) {
        return false;
    }
}
std::vector<CloudFile> MegaProvider::list_files(const std::string& folder_id) { return {}; }
std::string MegaProvider::create_folder(const std::string& name, const std::string& parent_id) { return ""; }

// Local Storage Implementation (Fully functional for testing)
LocalStorageProvider::LocalStorageProvider(const CloudStorageConfig& config)
    : CloudStorageProvider(config) {
    storage_root_ = config.compaction_directory;
    fs::create_directories(storage_root_);
}

UploadResult LocalStorageProvider::upload_file(const std::string& local_path,
                                             const std::string& remote_name,
                                             const std::unordered_map<std::string, std::string>& metadata) {
    UploadResult result;
    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        if (!fs::exists(local_path)) {
            result.success = false;
            result.error_message = "Local file does not exist: " + local_path;
            return result;
        }

        std::string remote_path = storage_root_ + "/" + remote_name;
        fs::copy_file(local_path, remote_path, fs::copy_options::overwrite_existing);

        auto file_size = fs::file_size(remote_path);
        std::string file_id = generate_unique_filename(remote_name);

        // Register the file
        CloudFile cloud_file;
        cloud_file.file_id = file_id;
        cloud_file.name = remote_name;
        cloud_file.path = remote_path;
        cloud_file.size_bytes = file_size;
        cloud_file.created_time = std::chrono::system_clock::now();
        cloud_file.modified_time = cloud_file.created_time;
        cloud_file.metadata = metadata;

        file_registry_[file_id] = cloud_file;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        result.success = true;
        result.file_id = file_id;
        result.uploaded_size = file_size;
        result.upload_time = duration;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("Upload failed: ") + e.what();
    }

    return result;
}

UploadResult LocalStorageProvider::upload_data(const std::vector<uint8_t>& data,
                                             const std::string& remote_name,
                                             const std::unordered_map<std::string, std::string>& metadata) {
    UploadResult result;
    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        std::string remote_path = storage_root_ + "/" + remote_name;
        std::ofstream file(remote_path, std::ios::binary);

        if (!file.is_open()) {
            result.success = false;
            result.error_message = "Cannot create file: " + remote_path;
            return result;
        }

        file.write(reinterpret_cast<const char*>(data.data()), data.size());
        file.close();

        std::string file_id = generate_unique_filename(remote_name);

        // Register the file
        CloudFile cloud_file;
        cloud_file.file_id = file_id;
        cloud_file.name = remote_name;
        cloud_file.path = remote_path;
        cloud_file.size_bytes = data.size();
        cloud_file.created_time = std::chrono::system_clock::now();
        cloud_file.modified_time = cloud_file.created_time;
        cloud_file.metadata = metadata;

        file_registry_[file_id] = cloud_file;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        result.success = true;
        result.file_id = file_id;
        result.uploaded_size = data.size();
        result.upload_time = duration;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("Upload failed: ") + e.what();
    }

    return result;
}

DownloadResult LocalStorageProvider::download_file(const std::string& file_id) {
    DownloadResult result;

    auto it = file_registry_.find(file_id);
    if (it == file_registry_.end()) {
        result.success = false;
        result.error_message = "File not found: " + file_id;
        return result;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        const CloudFile& cloud_file = it->second;
        std::ifstream file(cloud_file.path, std::ios::binary | std::ios::ate);

        if (!file.is_open()) {
            result.success = false;
            result.error_message = "Cannot open file: " + cloud_file.path;
            return result;
        }

        std::streamsize file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        result.data.resize(file_size);
        file.read(reinterpret_cast<char*>(result.data.data()), file_size);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        result.success = true;
        result.downloaded_size = file_size;
        result.download_time = duration;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("Download failed: ") + e.what();
    }

    return result;
}

bool LocalStorageProvider::delete_file(const std::string& file_id) {
    auto it = file_registry_.find(file_id);
    if (it == file_registry_.end()) {
        return false;
    }

    try {
        fs::remove(it->second.path);
        file_registry_.erase(it);
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

std::vector<CloudFile> LocalStorageProvider::list_files(const std::string& folder_id) {
    std::vector<CloudFile> files;

    for (const auto& pair : file_registry_) {
        files.push_back(pair.second);
    }

    return files;
}

std::string LocalStorageProvider::create_folder(const std::string& name, const std::string& parent_id) {
    std::string folder_path = storage_root_ + "/" + name;
    try {
        fs::create_directories(folder_path);
        return folder_path;
    } catch (const std::exception& e) {
        return "";
    }
}

// Cloud Storage Manager Implementation
CloudStorageManager::CloudStorageManager() : default_provider_(CloudProvider::LOCAL_STORAGE) {
    // Add local storage by default for testing
    CloudStorageConfig local_config;
    local_config.provider = CloudProvider::LOCAL_STORAGE;

    // Use environment variable for compaction directory
    if (const char* env_dir = std::getenv("HRM_CLOUD_STORAGE_DIR")) {
        local_config.compaction_directory = env_dir;
    } else {
        local_config.compaction_directory = "./cloud_storage";
    }

    local_config.compaction_folder_name = "compactions";
    local_config.auto_refresh_tokens = false;
    // Resource-aware timeout calculations for network operations
    auto resource_monitor = std::make_shared<ResourceMonitor>();
    auto upload_timeout = resource_monitor->calculate_process_timeout("network");
    auto download_timeout = resource_monitor->calculate_process_timeout("network");
    local_config.upload_timeout = upload_timeout;
    local_config.download_timeout = download_timeout;

    auto local_provider = std::make_shared<LocalStorageProvider>(local_config);
    providers_[CloudProvider::LOCAL_STORAGE] = local_provider;

    // Try to load cloud providers from environment variables
    load_providers_from_environment();
}

void CloudStorageManager::load_providers_from_environment() {
    // Load Google Drive configuration
    if (const char* api_key = std::getenv("HRM_CLOUD_API_KEY")) {
        CloudStorageConfig google_config;
        google_config.provider = CloudProvider::GOOGLE_DRIVE;
        google_config.api_key = api_key;

        if (const char* client_id = std::getenv("HRM_GOOGLE_CLIENT_ID")) {
            google_config.client_id = client_id;
        }
        if (const char* client_secret = std::getenv("HRM_GOOGLE_CLIENT_SECRET")) {
            google_config.client_secret = client_secret;
        }
        if (const char* refresh_token = std::getenv("HRM_GOOGLE_REFRESH_TOKEN")) {
            google_config.refresh_token = refresh_token;
        }

        google_config.compaction_directory = "./cloud_storage";
        google_config.compaction_folder_name = "compactions";
        google_config.auto_refresh_tokens = true;

        auto google_provider = std::make_shared<GoogleDriveProvider>(google_config);
        if (google_provider->validate_config()) {
            providers_[CloudProvider::GOOGLE_DRIVE] = google_provider;
        }
    }

    // Load Dropbox configuration
    if (const char* api_key = std::getenv("HRM_DROPBOX_API_KEY")) {
        CloudStorageConfig dropbox_config;
        dropbox_config.provider = CloudProvider::DROPBOX;
        dropbox_config.api_key = api_key;

        dropbox_config.compaction_directory = "./cloud_storage";
        dropbox_config.compaction_folder_name = "compactions";
        dropbox_config.auto_refresh_tokens = true;

        auto dropbox_provider = std::make_shared<DropboxProvider>(dropbox_config);
        if (dropbox_provider->validate_config()) {
            providers_[CloudProvider::DROPBOX] = dropbox_provider;
        }
    }
}

CloudStorageManager::~CloudStorageManager() {
    // Cleanup
}

bool CloudStorageManager::add_provider(std::shared_ptr<CloudStorageProvider> provider) {
    if (!provider) return false;

    CloudProvider type = provider->get_provider();
    providers_[type] = provider;
    return true;
}

bool CloudStorageManager::remove_provider(CloudProvider provider) {
    auto it = providers_.find(provider);
    if (it != providers_.end()) {
        providers_.erase(it);
        return true;
    }
    return false;
}

std::shared_ptr<CloudStorageProvider> CloudStorageManager::get_provider(CloudProvider provider) const {
    auto it = providers_.find(provider);
    return (it != providers_.end()) ? it->second : nullptr;
}

std::vector<CloudProvider> CloudStorageManager::get_available_providers() const {
    std::vector<CloudProvider> available;
    for (const auto& pair : providers_) {
        available.push_back(pair.first);
    }
    return available;
}

UploadResult CloudStorageManager::upload_compacted_memory(const std::string& compaction_id,
                                                        const std::vector<uint8_t>& data,
                                                        CloudProvider provider) {
    if (!validate_provider(provider)) {
        UploadResult result;
        result.success = false;
        result.error_message = "Invalid or unavailable provider";
        return result;
    }

    auto provider_ptr = providers_[provider];
    std::string filename = generate_compaction_filename(compaction_id);

    return provider_ptr->upload_data(data, filename);
}

DownloadResult CloudStorageManager::download_compacted_memory(const std::string& compaction_id,
                                                            CloudProvider provider) {
    if (!validate_provider(provider)) {
        DownloadResult result;
        result.success = false;
        result.error_message = "Invalid or unavailable provider";
        return result;
    }

    // For now, we need to find the file by name pattern
    // In a real implementation, this would be stored in a database
    auto provider_ptr = providers_[provider];
    auto files = provider_ptr->list_files();

    for (const auto& file : files) {
        if (file.name.find(compaction_id) != std::string::npos) {
            return provider_ptr->download_file(file.file_id);
        }
    }

    DownloadResult result;
    result.success = false;
    result.error_message = "Compaction file not found";
    return result;
}

bool CloudStorageManager::delete_compacted_memory(const std::string& compaction_id,
                                                CloudProvider provider) {
    if (!validate_provider(provider)) {
        return false;
    }

    auto provider_ptr = providers_[provider];
    auto files = provider_ptr->list_files();

    for (const auto& file : files) {
        if (file.name.find(compaction_id) != std::string::npos) {
            return provider_ptr->delete_file(file.file_id);
        }
    }

    return false;
}

std::vector<UploadResult> CloudStorageManager::upload_multiple_files(
    const std::vector<std::pair<std::string, std::string>>& files,
    CloudProvider provider) {

    std::vector<UploadResult> results;

    if (!validate_provider(provider)) {
        for (const auto& file_pair : files) {
            UploadResult result;
            result.success = false;
            result.error_message = "Invalid provider";
            results.push_back(result);
        }
        return results;
    }

    auto provider_ptr = providers_[provider];

    for (const auto& file_pair : files) {
        const std::string& local_path = file_pair.first;
        const std::string& remote_name = file_pair.second;

        auto result = provider_ptr->upload_file(local_path, remote_name);
        results.push_back(result);
    }

    return results;
}

std::unordered_map<CloudProvider, uint64_t> CloudStorageManager::get_storage_usage() const {
    std::unordered_map<CloudProvider, uint64_t> usage;

    for (const auto& pair : providers_) {
        CloudProvider provider = pair.first;
        auto provider_ptr = pair.second;

        uint64_t total_size = 0;
        auto files = provider_ptr->list_files();

        for (const auto& file : files) {
            total_size += file.size_bytes;
        }

        usage[provider] = total_size;
    }

    return usage;
}

std::unordered_map<CloudProvider, std::vector<CloudFile>> CloudStorageManager::get_all_files() const {
    std::unordered_map<CloudProvider, std::vector<CloudFile>> all_files;

    for (const auto& pair : providers_) {
        CloudProvider provider = pair.first;
        auto provider_ptr = pair.second;

        all_files[provider] = provider_ptr->list_files();
    }

    return all_files;
}

void CloudStorageManager::set_default_provider(CloudProvider provider) {
    if (validate_provider(provider)) {
        default_provider_ = provider;
    }
}

CloudProvider CloudStorageManager::get_default_provider() const {
    return default_provider_;
}

std::string CloudStorageManager::generate_compaction_filename(const std::string& compaction_id) const {
    return "compaction_" + compaction_id + ".mem";
}

bool CloudStorageManager::validate_provider(CloudProvider provider) const {
    auto it = providers_.find(provider);
    return it != providers_.end() && it->second != nullptr;
}