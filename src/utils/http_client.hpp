#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

struct HTTPResponse {
    int status_code;
    std::string status_message;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
    bool success;
    std::string error_message;
};

struct HTTPRequest {
    std::string method;
    std::string url;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
    int timeout_seconds; // 0 = resource-aware calculation
};

class HTTPClient {
public:
    HTTPClient();
    ~HTTPClient();

    HTTPResponse request(const HTTPRequest& req);
    HTTPResponse get(const std::string& url, const std::unordered_map<std::string, std::string>& headers = {});
    HTTPResponse post(const std::string& url, const std::string& body, const std::unordered_map<std::string, std::string>& headers = {});
    HTTPResponse put(const std::string& url, const std::string& body, const std::unordered_map<std::string, std::string>& headers = {});
    HTTPResponse patch(const std::string& url, const std::string& body, const std::unordered_map<std::string, std::string>& headers = {});
    HTTPResponse delete_request(const std::string& url, const std::unordered_map<std::string, std::string>& headers = {});

private:
    // Socket-based HTTP implementation
    int create_socket(const std::string& host, int port);
    bool connect_socket(int sockfd, const std::string& host, int port);
    std::string send_request(int sockfd, const std::string& request);
    HTTPResponse parse_response(const std::string& response);
    std::string url_encode(const std::string& str);
    void parse_url(const std::string& url, std::string& host, std::string& path, int& port, bool& is_https);
};