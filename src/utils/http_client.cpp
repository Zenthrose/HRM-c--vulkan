#include "http_client.hpp"
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cctype>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#endif

HTTPClient::HTTPClient() {
#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
}

HTTPClient::~HTTPClient() {
#ifdef _WIN32
    WSACleanup();
#endif
}

HTTPResponse HTTPClient::request(const HTTPRequest& req) {
    HTTPResponse response;
    response.success = false;

    std::string host, path;
    int port;
    bool is_https;
    parse_url(req.url, host, path, port, is_https);

    if (is_https) {
        response.error_message = "HTTPS not supported";
        return response;
    }

    int sockfd = create_socket(host, port);
    if (sockfd < 0) {
        response.error_message = "Failed to create socket";
        return response;
    }

    if (!connect_socket(sockfd, host, port)) {
        response.error_message = "Failed to connect";
#ifdef _WIN32
        closesocket(sockfd);
#else
        close(sockfd);
#endif
        return response;
    }

    std::string http_request = req.method + " " + path + " HTTP/1.1\r\n";
    http_request += "Host: " + host + "\r\n";
    for (const auto& header : req.headers) {
        http_request += header.first + ": " + header.second + "\r\n";
    }
    if (!req.body.empty()) {
        http_request += "Content-Length: " + std::to_string(req.body.size()) + "\r\n";
    }
    http_request += "\r\n" + req.body;

    std::string raw_response = send_request(sockfd, http_request);
    if (raw_response.empty()) {
        response.error_message = "Failed to send request or receive response";
#ifdef _WIN32
        closesocket(sockfd);
#else
        close(sockfd);
#endif
        return response;
    }

    response = parse_response(raw_response);
    response.success = (response.status_code >= 200 && response.status_code < 300);

#ifdef _WIN32
    closesocket(sockfd);
#else
    close(sockfd);
#endif

    return response;
}

HTTPResponse HTTPClient::get(const std::string& url, const std::unordered_map<std::string, std::string>& headers) {
    HTTPRequest req;
    req.method = "GET";
    req.url = url;
    req.headers = headers;
    return request(req);
}

HTTPResponse HTTPClient::post(const std::string& url, const std::string& body, const std::unordered_map<std::string, std::string>& headers) {
    HTTPRequest req;
    req.method = "POST";
    req.url = url;
    req.body = body;
    req.headers = headers;
    return request(req);
}

HTTPResponse HTTPClient::put(const std::string& url, const std::string& body, const std::unordered_map<std::string, std::string>& headers) {
    HTTPRequest req;
    req.method = "PUT";
    req.url = url;
    req.body = body;
    req.headers = headers;
    return request(req);
}

HTTPResponse HTTPClient::patch(const std::string& url, const std::string& body, const std::unordered_map<std::string, std::string>& headers) {
    HTTPRequest req;
    req.method = "PATCH";
    req.url = url;
    req.body = body;
    req.headers = headers;
    return request(req);
}

HTTPResponse HTTPClient::delete_request(const std::string& url, const std::unordered_map<std::string, std::string>& headers) {
    HTTPRequest req;
    req.method = "DELETE";
    req.url = url;
    req.headers = headers;
    return request(req);
}

int HTTPClient::create_socket(const std::string& host, int port) {
#ifdef _WIN32
    return socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
#else
    return socket(AF_INET, SOCK_STREAM, 0);
#endif
}

bool HTTPClient::connect_socket(int sockfd, const std::string& host, int port) {
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);

    struct hostent* he = gethostbyname(host.c_str());
    if (!he) return false;

    memcpy(&server_addr.sin_addr, he->h_addr, he->h_length);

    return connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == 0;
}

std::string HTTPClient::send_request(int sockfd, const std::string& request) {
    if (send(sockfd, request.c_str(), request.size(), 0) < 0) {
        return "";
    }

    std::string response;
    char buffer[4096];
    int bytes_received;
    while ((bytes_received = recv(sockfd, buffer, sizeof(buffer) - 1, 0)) > 0) {
        buffer[bytes_received] = '\0';
        response += buffer;
        // Simple check for end of response (not perfect)
        if (response.find("\r\n\r\n") != std::string::npos && response.size() > 100) break;
    }
    return response;
}

HTTPResponse HTTPClient::parse_response(const std::string& response) {
    HTTPResponse res;
    std::istringstream iss(response);
    std::string line;

    // Parse status line
    if (std::getline(iss, line)) {
        std::istringstream status_iss(line);
        std::string http_version;
        status_iss >> http_version >> res.status_code;
        std::getline(status_iss, res.status_message);
        // Remove leading space
        if (!res.status_message.empty() && res.status_message[0] == ' ') {
            res.status_message = res.status_message.substr(1);
        }
    }

    // Parse headers
    while (std::getline(iss, line) && line != "\r") {
        size_t colon_pos = line.find(':');
        if (colon_pos != std::string::npos) {
            std::string key = line.substr(0, colon_pos);
            std::string value = line.substr(colon_pos + 1);
            // Trim spaces
            key.erase(key.begin(), std::find_if(key.begin(), key.end(), [](int ch) { return !std::isspace(ch); }));
            key.erase(std::find_if(key.rbegin(), key.rend(), [](int ch) { return !std::isspace(ch); }).base(), key.end());
            value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](int ch) { return !std::isspace(ch); }));
            value.erase(std::find_if(value.rbegin(), value.rend(), [](int ch) { return !std::isspace(ch); }).base(), value.end());
            res.headers[key] = value;
        }
    }

    // Body
    while (std::getline(iss, line)) {
        res.body += line + "\n";
    }
    if (!res.body.empty()) res.body.pop_back(); // Remove last \n

    return res;
}

std::string HTTPClient::url_encode(const std::string& str) {
    std::string encoded;
    for (char c : str) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            encoded += c;
        } else {
            char buf[4];
            sprintf(buf, "%%%02X", (unsigned char)c);
            encoded += buf;
        }
    }
    return encoded;
}

void HTTPClient::parse_url(const std::string& url, std::string& host, std::string& path, int& port, bool& is_https) {
    size_t protocol_end = url.find("://");
    if (protocol_end != std::string::npos) {
        std::string protocol = url.substr(0, protocol_end);
        is_https = (protocol == "https");
        size_t host_start = protocol_end + 3;
        size_t port_start = url.find(':', host_start);
        size_t path_start = url.find('/', host_start);

        if (port_start != std::string::npos && (path_start == std::string::npos || port_start < path_start)) {
            host = url.substr(host_start, port_start - host_start);
            port = std::stoi(url.substr(port_start + 1, path_start - port_start - 1));
        } else {
            host = url.substr(host_start, path_start - host_start);
            port = is_https ? 443 : 80;
        }

        if (path_start != std::string::npos) {
            path = url.substr(path_start);
        } else {
            path = "/";
        }
    } else {
        // Assume http
        is_https = false;
        size_t host_start = 0;
        size_t path_start = url.find('/');
        host = url.substr(0, path_start);
        port = 80;
        path = (path_start != std::string::npos) ? url.substr(path_start) : "/";
    }
}