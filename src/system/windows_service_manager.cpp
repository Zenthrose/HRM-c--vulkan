#include "windows_service_manager.hpp"
#include <iostream>
#include <sstream>

WindowsServiceManager* WindowsServiceManager::instance_ = nullptr;

WindowsServiceManager::WindowsServiceManager() 
    : status_handle_(nullptr), is_running_(false),
      idle_check_interval_(std::chrono::milliseconds(1000)),
      idle_threshold_(std::chrono::seconds(300)), // 5 minutes idle threshold
      last_input_time_(0), auto_boot_enabled_(true), idle_processing_enabled_(true) {
    
    // Initialize service status
    service_status_.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
    service_status_.dwControlsAccepted = SERVICE_ACCEPT_STOP | SERVICE_ACCEPT_SHUTDOWN;
    service_status_.dwWin32ExitCode = 0;
    service_status_.dwServiceSpecificExitCode = 0;
    service_status_.dwCheckPoint = 0;
    service_status_.dwWaitHint = 0;
    
    // Create stop event
    stop_event_ = CreateEvent(nullptr, TRUE, FALSE, nullptr);
    if (!stop_event_) {
        throw std::runtime_error("Failed to create stop event");
    }
    
    std::cout << "Windows Service Manager initialized" << std::endl;
}

WindowsServiceManager::~WindowsServiceManager() {
    if (stop_event_) {
        CloseHandle(stop_event_);
    }
    
    stop_service();
}

bool WindowsServiceManager::install_service() {
    std::cout << "Installing Windows service..." << std::endl;
    
    // Get current executable path
    char exe_path[MAX_PATH];
    GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
    
    // Open service manager
    SC_HANDLE scm = OpenSCManager(nullptr, nullptr, SC_MANAGER_ALL_ACCESS);
    if (!scm) {
        log_event("Failed to open service manager", EVENTLOG_ERROR_TYPE);
        return false;
    }
    
    // Create service
    SC_HANDLE service = CreateServiceA(
        scm,
        "HRMSystem",
        "HRM Self-Evolving AI System",
        SERVICE_ALL_ACCESS,
        SERVICE_WIN32_OWN_PROCESS,
        SERVICE_AUTO_START,
        SERVICE_ERROR_NORMAL,
        exe_path,
        nullptr,
        0,
        nullptr,
        nullptr,
        nullptr
    );
    
    if (!service) {
        DWORD error = GetLastError();
        std::ostringstream msg;
        msg << "Failed to create service. Error: " << error;
        log_event(msg.str(), EVENTLOG_ERROR_TYPE);
        CloseServiceHandle(scm);
        return false;
    }
    
    // Set service description
    SERVICE_DESCRIPTIONA desc;
    desc.lpDescription = const_cast<char*>("HRM Self-Evolving AI System with auto-boot and idle processing capabilities");
    ChangeServiceConfig2A(service, SERVICE_CONFIG_DESCRIPTION, &desc);
    
    CloseServiceHandle(service);
    CloseServiceHandle(scm);
    
    log_event("Windows service installed successfully", EVENTLOG_INFORMATION_TYPE);
    return true;
}

bool WindowsServiceManager::uninstall_service() {
    std::cout << "Uninstalling Windows service..." << std::endl;
    
    SC_HANDLE scm = OpenSCManager(nullptr, nullptr, SC_MANAGER_ALL_ACCESS);
    if (!scm) {
        log_event("Failed to open service manager for uninstall", EVENTLOG_ERROR_TYPE);
        return false;
    }
    
    SC_HANDLE service = OpenServiceA(scm, "HRMSystem", SERVICE_ALL_ACCESS);
    if (!service) {
        log_event("Failed to open service for uninstall", EVENTLOG_ERROR_TYPE);
        CloseServiceHandle(scm);
        return false;
    }
    
    // Stop service if running
    SERVICE_STATUS status;
    if (QueryServiceStatus(service, &status)) {
        if (status.dwCurrentState == SERVICE_RUNNING) {
            ControlService(service, SERVICE_CONTROL_STOP, &status);
            // Wait for service to stop
            int timeout = 10; // 10 seconds
            while (status.dwCurrentState != SERVICE_STOPPED && timeout > 0) {
                Sleep(1000);
                QueryServiceStatus(service, &status);
                timeout--;
            }
        }
    }
    
    // Delete service
    bool deleted = DeleteService(service);
    
    CloseServiceHandle(service);
    CloseServiceHandle(scm);
    
    if (deleted) {
        log_event("Windows service uninstalled successfully", EVENTLOG_INFORMATION_TYPE);
    } else {
        log_event("Failed to uninstall Windows service", EVENTLOG_ERROR_TYPE);
    }
    
    return deleted;
}

bool WindowsServiceManager::start_service() {
    std::cout << "Starting Windows service..." << std::endl;
    
    SC_HANDLE scm = OpenSCManager(nullptr, nullptr, SC_MANAGER_ALL_ACCESS);
    if (!scm) {
        log_event("Failed to open service manager for start", EVENTLOG_ERROR_TYPE);
        return false;
    }
    
    SC_HANDLE service = OpenServiceA(scm, "HRMSystem", SERVICE_ALL_ACCESS);
    if (!service) {
        log_event("Failed to open service for start", EVENTLOG_ERROR_TYPE);
        CloseServiceHandle(scm);
        return false;
    }
    
    // Start service
    bool started = StartServiceA(service, 0, nullptr);
    
    CloseServiceHandle(service);
    CloseServiceHandle(scm);
    
    if (started) {
        log_event("Windows service started successfully", EVENTLOG_INFORMATION_TYPE);
    } else {
        log_event("Failed to start Windows service", EVENTLOG_ERROR_TYPE);
    }
    
    return started;
}

bool WindowsServiceManager::stop_service() {
    std::cout << "Stopping Windows service..." << std::endl;
    
    is_running_ = false;
    
    // Signal threads to stop
    if (stop_event_) {
        SetEvent(stop_event_);
    }
    
    // Wait for threads to finish
    if (idle_monitor_thread_.joinable()) {
        idle_monitor_thread_.join();
    }
    
    if (auto_boot_thread_.joinable()) {
        auto_boot_thread_.join();
    }
    
    return true;
}

void WindowsServiceManager::add_boot_task(const std::function<void()>& task) {
    std::lock_guard<std::mutex> guard(task_mutex_);
    boot_tasks_.push_back(task);
    std::cout << "Added boot task. Total: " << boot_tasks_.size() << std::endl;
}

void WindowsServiceManager::add_idle_task(const std::function<void()>& task) {
    std::lock_guard<std::mutex> guard(task_mutex_);
    idle_tasks_.push_back(task);
    std::cout << "Added idle task. Total: " << idle_tasks_.size() << std::endl;
}

void WindowsServiceManager::set_auto_boot(bool enabled) {
    auto_boot_enabled_ = enabled;
    std::cout << "Auto-boot " << (enabled ? "enabled" : "disabled") << std::endl;
}

void WindowsServiceManager::set_idle_processing(bool enabled) {
    idle_processing_enabled_ = enabled;
    std::cout << "Idle processing " << (enabled ? "enabled" : "disabled") << std::endl;
}

void WindowsServiceManager::set_idle_threshold(std::chrono::seconds threshold) {
    idle_threshold_ = threshold;
    std::cout << "Idle threshold set to " << threshold.count() << " seconds" << std::endl;
}

bool WindowsServiceManager::is_service_running() const {
    return is_running_;
}

std::chrono::seconds WindowsServiceManager::get_idle_duration() const {
    DWORD current_time = get_last_input_time();
    DWORD idle_time = current_time - last_input_time_;
    return std::chrono::seconds(idle_time / 1000); // Convert from milliseconds
}

void WindowsServiceManager::service_main() {
    std::cout << "HRM Windows service starting..." << std::endl;
    
    // Register service control handler
    status_handle_ = RegisterServiceCtrlHandlerA("HRMSystem", WindowsServiceManager::service_control_handler);
    if (!status_handle_) {
        log_event("Failed to register service control handler", EVENTLOG_ERROR_TYPE);
        return;
    }
    
    // Update service status to starting
    update_service_status(SERVICE_START_PENDING);
    
    // Initialize service components
    is_running_ = true;
    last_input_time_ = get_last_input_time();
    
    // Start auto-boot tasks
    if (auto_boot_enabled_) {
        auto_boot_thread_ = std::thread(&WindowsServiceManager::execute_boot_tasks, this);
    }
    
    // Start idle monitoring
    if (idle_processing_enabled_) {
        idle_monitor_thread_ = std::thread(&WindowsServiceManager::monitor_idle_time, this);
    }
    
    // Update service status to running
    update_service_status(SERVICE_RUNNING);
    
    log_event("HRM Windows service started successfully", EVENTLOG_INFORMATION_TYPE);
    
    // Wait for stop signal
    WaitForSingleObject(stop_event_, INFINITE);
    
    // Cleanup
    is_running_ = false;
    update_service_status(SERVICE_STOPPED);
    
    log_event("HRM Windows service stopped", EVENTLOG_INFORMATION_TYPE);
}

void WindowsServiceManager::update_service_status(DWORD current_state, DWORD exit_code) {
    service_status_.dwCurrentState = current_state;
    service_status_.dwWin32ExitCode = exit_code;
    
    SetServiceStatus(status_handle_, &service_status_);
}

void WindowsServiceManager::monitor_idle_time() {
    while (is_running_) {
        DWORD current_time = get_last_input_time();
        DWORD idle_duration = current_time - last_input_time_;
        
        if (idle_duration >= (idle_threshold_.count() * 1000)) { // Convert to milliseconds
            if (!idle_tasks_.empty()) {
                execute_idle_tasks();
            }
        }
        
        // Update last input time if there's activity
        if (idle_duration < 1000) { // Less than 1 second since last check
            last_input_time_ = current_time;
        }
        
        // Wait for next check
        std::this_thread::sleep_for(idle_check_interval_);
    }
}

void WindowsServiceManager::execute_boot_tasks() {
    std::cout << "Executing " << boot_tasks_.size() << " boot tasks..." << std::endl;
    
    for (const auto& task : boot_tasks_) {
        try {
            task();
            log_event("Boot task executed successfully", EVENTLOG_INFORMATION_TYPE);
        } catch (const std::exception& e) {
            std::ostringstream msg;
            msg << "Boot task failed: " << e.what();
            log_event(msg.str(), EVENTLOG_ERROR_TYPE);
        }
    }
    
    log_event("All boot tasks completed", EVENTLOG_INFORMATION_TYPE);
}

void WindowsServiceManager::execute_idle_tasks() {
    std::cout << "Executing " << idle_tasks_.size() << " idle tasks..." << std::endl;
    
    for (const auto& task : idle_tasks_) {
        try {
            task();
            log_event("Idle task executed successfully", EVENTLOG_INFORMATION_TYPE);
        } catch (const std::exception& e) {
            std::ostringstream msg;
            msg << "Idle task failed: " << e.what();
            log_event(msg.str(), EVENTLOG_ERROR_TYPE);
        }
    }
    
    log_event("All idle tasks completed", EVENTLOG_INFORMATION_TYPE);
}

DWORD WindowsServiceManager::get_last_input_time() const {
    // Get last input time from Windows
    LASTINPUTINFO lii;
    lii.cbSize = sizeof(LASTINPUTINFO);
    
    if (GetLastInputInfo(&lii)) {
        return lii.dwTime;
    }
    
    return GetTickCount();
}

bool WindowsServiceManager::is_system_idle() const {
    DWORD current_time = get_last_input_time();
    DWORD idle_duration = current_time - last_input_time_;
    return idle_duration >= (idle_threshold_.count() * 1000);
}

void WindowsServiceManager::log_event(const std::string& message, WORD type) {
    // Log to Windows Event Log
    HANDLE hEventSource = RegisterEventSourceA(nullptr, "HRMSystem");
    
    if (hEventSource) {
        const char* messages[] = { message.c_str() };
        ReportEventA(hEventSource, type, 0, 0, nullptr, 1, 0, messages, nullptr);
        DeregisterEventSource(hEventSource);
    }
    
    // Also log to console for debugging
    std::cout << "[EVENT] " << message << std::endl;
}

void WINAPI WindowsServiceManager::service_entry(DWORD argc, LPSTR* argv) {
    if (!instance_) {
        return;
    }
    
    instance_->service_main();
}

void WINAPI WindowsServiceManager::service_control_handler(DWORD control_code) {
    if (!instance_) {
        return;
    }
    
    switch (control_code) {
        case SERVICE_CONTROL_STOP:
        case SERVICE_CONTROL_SHUTDOWN:
            instance_->update_service_status(SERVICE_STOP_PENDING);
            SetEvent(instance_->stop_event_);
            break;
            
        case SERVICE_CONTROL_INTERROGATE:
            instance_->update_service_status(instance_->service_status_.dwCurrentState);
            break;
            
        default:
            break;
    }
}

// Static member initialization