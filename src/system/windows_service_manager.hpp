#pragma once

#include <windows.h>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <memory>
#include <vector>
#include <functional>
#include <mutex>

class WindowsServiceManager {
private:
    SERVICE_STATUS_HANDLE status_handle_;
    SERVICE_STATUS service_status_;
    HANDLE stop_event_;
    std::atomic<bool> is_running_;
    std::thread idle_monitor_thread_;
    std::thread auto_boot_thread_;
    std::vector<std::function<void()>> idle_tasks_;
    std::vector<std::function<void()>> boot_tasks_;
    std::chrono::milliseconds idle_check_interval_;
    std::chrono::seconds idle_threshold_;
    DWORD last_input_time_;
    bool auto_boot_enabled_;
    bool idle_processing_enabled_;
    std::mutex task_mutex_;
    
    void service_main();
    void update_service_status(DWORD current_state, DWORD exit_code = 0);
    void monitor_idle_time();
    void execute_boot_tasks();
    void execute_idle_tasks();
    DWORD get_last_input_time() const;
    bool is_system_idle() const;
    void log_event(const std::string& message, WORD type = EVENTLOG_INFORMATION_TYPE);
    static void WINAPI service_control_handler(DWORD control_code);

public:
    WindowsServiceManager();
    ~WindowsServiceManager();
    
    // Service lifecycle
    bool install_service();
    bool uninstall_service();
    bool start_service();
    bool stop_service();
    
    // Auto-boot configuration
    void add_boot_task(const std::function<void()>& task);
    void remove_boot_task(size_t index);
    void set_auto_boot(bool enabled);
    
    // Idle processing configuration
    void add_idle_task(const std::function<void()>& task);
    void remove_idle_task(size_t index);
    void set_idle_threshold(std::chrono::seconds threshold);
    void set_idle_check_interval(std::chrono::milliseconds interval);
    void set_idle_processing(bool enabled);
    
    // Status and monitoring
    bool is_service_running() const;
    std::chrono::seconds get_idle_duration() const;
    std::string get_service_status() const;
    void print_service_info() const;
    
    // Service entry point
    static void WINAPI service_entry(DWORD argc, LPSTR* argv);
    static WindowsServiceManager* instance_;
};