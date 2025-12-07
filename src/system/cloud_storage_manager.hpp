#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

enum class CloudProvider {
    GOOGLE_DRIVE,
    DROPBOX,
    ONEDRIVE,
    MEGA,
    LOCAL_STORAGE // For testing/development
};

enum class UploadStatus {
    PENDING,
    UPLOADING,
    COMPLETED,
    FAILED,
    CANCELLED
};

struct CloudFile {
    std::string file_id;
    std::string name;
    std::string path;
    uint64_t size_bytes;
    std::chrono::system_clock::time_point created_time;
    std::chrono::system_clock::time_point modified_time;
    std::string download_url;
    std::unordered_map<std::string, std::string> metadata;
};

struct UploadResult {
    bool success;
    std::string file_id;
    std::string download_url;
    uint64_t uploaded_size;
    std::chrono::milliseconds upload_time;
    std::string error_message;
};

struct DownloadResult {
    bool success;
    std::vector<uint8_t> data;
    uint64_t downloaded_size;
    std::chrono::milliseconds download_time;
    std::string error_message;
};

struct CloudStorageConfig {
    CloudProvider provider;
    std::string api_key;
    std::string client_id;
    std::string client_secret;
    std::string refresh_token;
    std::string access_token;
    std::chrono::seconds token_expiry;
    std::string root_folder_id;
    std::string compaction_folder_name;
    std::string compaction_directory;
    bool auto_refresh_tokens;
    std::chrono::seconds upload_timeout;
    std::chrono::seconds download_timeout;
};

class CloudStorageProvider {
public:
    CloudStorageProvider(const CloudStorageConfig& config);
    virtual ~CloudStorageProvider() = default;

    // Authentication
    virtual bool authenticate() = 0;
    virtual bool refresh_token() = 0;
    virtual bool is_authenticated() const = 0;

    // File operations
    virtual UploadResult upload_file(const std::string& local_path,
                                   const std::string& remote_name,
                                   const std::unordered_map<std::string, std::string>& metadata = {}) = 0;

    virtual UploadResult upload_data(const std::vector<uint8_t>& data,
                                   const std::string& remote_name,
                                   const std::unordered_map<std::string, std::string>& metadata = {}) = 0;

    virtual DownloadResult download_file(const std::string& file_id) = 0;
    virtual bool delete_file(const std::string& file_id) = 0;

    // Directory operations
    virtual std::vector<CloudFile> list_files(const std::string& folder_id = "") = 0;
    virtual std::string create_folder(const std::string& name, const std::string& parent_id = "") = 0;

    // Provider info
    CloudProvider get_provider() const { return config_.provider; }
    std::string get_provider_name() const;

    // Validation
    bool validate_config() const;

protected:
    CloudStorageConfig config_;
    std::chrono::system_clock::time_point last_auth_time_;

    // Helper methods
    std::string generate_unique_filename(const std::string& base_name);
    std::string url_encode(const std::string& str) const;
    std::string base64_encode(const std::vector<uint8_t>& data) const;
};

// Google Drive implementation
class GoogleDriveProvider : public CloudStorageProvider {
public:
    GoogleDriveProvider(const CloudStorageConfig& config);

    bool authenticate() override;
    bool refresh_token() override;
    bool is_authenticated() const override;

    UploadResult upload_file(const std::string& local_path,
                           const std::string& remote_name,
                           const std::unordered_map<std::string, std::string>& metadata = {}) override;

    UploadResult upload_data(const std::vector<uint8_t>& data,
                           const std::string& remote_name,
                           const std::unordered_map<std::string, std::string>& metadata = {}) override;

    DownloadResult download_file(const std::string& file_id) override;
    bool delete_file(const std::string& file_id) override;

    std::vector<CloudFile> list_files(const std::string& folder_id = "") override;
    std::string create_folder(const std::string& name, const std::string& parent_id = "") override;

private:
    std::string api_base_url_;
    std::string upload_url_;

    // Google Drive specific methods
    std::string get_folder_id(const std::string& folder_name);
    std::string make_api_request(const std::string& url, const std::string& method = "GET",
                               const std::string& data = "", const std::string& content_type = "");
};

// Dropbox implementation
class DropboxProvider : public CloudStorageProvider {
public:
    DropboxProvider(const CloudStorageConfig& config);

    bool authenticate() override;
    bool refresh_token() override;
    bool is_authenticated() const override;

    UploadResult upload_file(const std::string& local_path,
                           const std::string& remote_name,
                           const std::unordered_map<std::string, std::string>& metadata = {}) override;

    UploadResult upload_data(const std::vector<uint8_t>& data,
                           const std::string& remote_name,
                           const std::unordered_map<std::string, std::string>& metadata = {}) override;

    DownloadResult download_file(const std::string& file_id) override;
    bool delete_file(const std::string& file_id) override;

    std::vector<CloudFile> list_files(const std::string& folder_id = "") override;
    std::string create_folder(const std::string& name, const std::string& parent_id = "") override;

private:
    std::string api_base_url_;
    std::string content_url_;

    std::string make_api_request(const std::string& endpoint, const std::string& method = "GET",
                               const std::string& data = "", const std::string& content_type = "");
};

// OneDrive implementation
class OneDriveProvider : public CloudStorageProvider {
public:
    OneDriveProvider(const CloudStorageConfig& config);

    bool authenticate() override;
    bool refresh_token() override;
    bool is_authenticated() const override;

    UploadResult upload_file(const std::string& local_path,
                           const std::string& remote_name,
                           const std::unordered_map<std::string, std::string>& metadata = {}) override;

    UploadResult upload_data(const std::vector<uint8_t>& data,
                           const std::string& remote_name,
                           const std::unordered_map<std::string, std::string>& metadata = {}) override;

    DownloadResult download_file(const std::string& file_id) override;
    bool delete_file(const std::string& file_id) override;

    std::vector<CloudFile> list_files(const std::string& folder_id = "") override;
    std::string create_folder(const std::string& name, const std::string& parent_id = "") override;

private:
    std::string api_base_url_;

    std::string make_api_request(const std::string& endpoint, const std::string& method = "GET",
                               const std::string& data = "", const std::string& content_type = "");
};

// Mega implementation
class MegaProvider : public CloudStorageProvider {
public:
    MegaProvider(const CloudStorageConfig& config);

    bool authenticate() override;
    bool refresh_token() override;
    bool is_authenticated() const override;

    UploadResult upload_file(const std::string& local_path,
                           const std::string& remote_name,
                           const std::unordered_map<std::string, std::string>& metadata = {}) override;

    UploadResult upload_data(const std::vector<uint8_t>& data,
                           const std::string& remote_name,
                           const std::unordered_map<std::string, std::string>& metadata = {}) override;

    DownloadResult download_file(const std::string& file_id) override;
    bool delete_file(const std::string& file_id) override;

    std::vector<CloudFile> list_files(const std::string& folder_id = "") override;
    std::string create_folder(const std::string& name, const std::string& parent_id = "") override;

private:
    // Mega-specific implementation would go here
    // Mega uses a different API structure
};

// Local storage for testing
class LocalStorageProvider : public CloudStorageProvider {
public:
    LocalStorageProvider(const CloudStorageConfig& config);

    bool authenticate() override { return true; }
    bool refresh_token() override { return true; }
    bool is_authenticated() const override { return true; }

    UploadResult upload_file(const std::string& local_path,
                           const std::string& remote_name,
                           const std::unordered_map<std::string, std::string>& metadata = {}) override;

    UploadResult upload_data(const std::vector<uint8_t>& data,
                           const std::string& remote_name,
                           const std::unordered_map<std::string, std::string>& metadata = {}) override;

    DownloadResult download_file(const std::string& file_id) override;
    bool delete_file(const std::string& file_id) override;

    std::vector<CloudFile> list_files(const std::string& folder_id = "") override;
    std::string create_folder(const std::string& name, const std::string& parent_id = "") override;

private:
    std::string storage_root_;
    std::unordered_map<std::string, CloudFile> file_registry_;
};

// Main cloud storage manager
class CloudStorageManager {
public:
    CloudStorageManager();
    ~CloudStorageManager();

    // Provider management
    bool add_provider(std::shared_ptr<CloudStorageProvider> provider);
    bool remove_provider(CloudProvider provider);
    std::shared_ptr<CloudStorageProvider> get_provider(CloudProvider provider) const;
    std::vector<CloudProvider> get_available_providers() const;

    // Memory compaction integration
    UploadResult upload_compacted_memory(const std::string& compaction_id,
                                       const std::vector<uint8_t>& data,
                                       CloudProvider provider = CloudProvider::LOCAL_STORAGE);

    DownloadResult download_compacted_memory(const std::string& compaction_id,
                                           CloudProvider provider = CloudProvider::LOCAL_STORAGE);

    bool delete_compacted_memory(const std::string& compaction_id,
                               CloudProvider provider = CloudProvider::LOCAL_STORAGE);

    // Batch operations
    std::vector<UploadResult> upload_multiple_files(const std::vector<std::pair<std::string, std::string>>& files,
                                                  CloudProvider provider = CloudProvider::LOCAL_STORAGE);

    // Storage management
    std::unordered_map<CloudProvider, uint64_t> get_storage_usage() const;
    std::unordered_map<CloudProvider, std::vector<CloudFile>> get_all_files() const;

    // Configuration
    void set_default_provider(CloudProvider provider);
    CloudProvider get_default_provider() const;

private:
    std::unordered_map<CloudProvider, std::shared_ptr<CloudStorageProvider>> providers_;
    CloudProvider default_provider_;

    // Helper methods
    std::string generate_compaction_filename(const std::string& compaction_id) const;
    bool validate_provider(CloudProvider provider) const;
    void load_providers_from_environment();
};