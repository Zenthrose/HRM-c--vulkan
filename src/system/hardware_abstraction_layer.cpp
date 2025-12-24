#include "hardware_abstraction_layer.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include "../vulkan/vulkan_instance_manager.hpp"
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <unistd.h>
#include <sys/sysinfo.h>
#include <sys/statvfs.h>
#endif
#ifndef NO_VULKAN
#include <vulkan/vulkan.h>
#endif

HardwareAbstractionLayer::HardwareAbstractionLayer()
    : current_backend_(ComputeBackend::AUTO), running_(false) {

    resource_monitor_ = std::make_shared<ResourceMonitor>();
    hardware_profile_ = detect_hardware();

    // Initialize active task counters
    active_tasks_per_backend_[ComputeBackend::GPU_VULKAN] = 0;
    active_tasks_per_backend_[ComputeBackend::CPU_PARALLEL] = 0;
    active_tasks_per_backend_[ComputeBackend::CPU_SEQUENTIAL] = 0;

    std::cout << "Hardware Abstraction Layer initialized" << std::endl;
    std::cout << "Detected hardware: " << hardware_profile_.gpu_name
              << " GPU, " << hardware_profile_.cpu_cores << " CPU cores" << std::endl;
}

HardwareAbstractionLayer::~HardwareAbstractionLayer() {
    running_ = false;
    if (task_processor_thread_.joinable()) {
        task_processor_thread_.join();
    }
    if (health_monitor_thread_.joinable()) {
        health_monitor_thread_.join();
    }
    cleanup_backends();
}

HardwareProfile HardwareAbstractionLayer::detect_hardware() {
    HardwareProfile profile;

    // Detect GPU capabilities
    auto gpu_profile = detect_gpu_capabilities();
    profile.gpu_name = gpu_profile.gpu_name;
    profile.gpu_memory_mb = gpu_profile.gpu_memory_mb;
    profile.gpu_compute_units = gpu_profile.gpu_compute_units;
    profile.vulkan_supported = gpu_profile.vulkan_supported;
    profile.cuda_supported = gpu_profile.cuda_supported;

    // Detect CPU capabilities
    auto cpu_profile = detect_cpu_capabilities();
    profile.cpu_name = cpu_profile.cpu_name;
    profile.cpu_cores = cpu_profile.cpu_cores;

    // Detect system capabilities
    auto sys_profile = detect_system_capabilities();
    profile.system_memory_mb = sys_profile.system_memory_mb;
    profile.available_storage_mb = sys_profile.available_storage_mb;

    // Determine capabilities
    if (profile.vulkan_supported && profile.gpu_memory_mb > 1024) {
        profile.capabilities.push_back(HardwareCapability::GPU_COMPUTE);
    }
    if (profile.cpu_cores >= 4) {
        profile.capabilities.push_back(HardwareCapability::CPU_MULTITHREADING);
    }
    if (profile.system_memory_mb >= 8192) {
        profile.capabilities.push_back(HardwareCapability::LARGE_MEMORY);
    }
    if (profile.available_storage_mb >= 1024 * 1024) { // 1TB
        profile.capabilities.push_back(HardwareCapability::FAST_STORAGE);
    }
    profile.capabilities.push_back(HardwareCapability::NETWORK_ACCESS);

    return profile;
}

bool HardwareAbstractionLayer::is_gpu_available() {
    return hardware_profile_.vulkan_supported && !hardware_profile_.gpu_name.empty();
}

bool HardwareAbstractionLayer::is_vulkan_supported() {
    return hardware_profile_.vulkan_supported;
}

void HardwareAbstractionLayer::update_hardware_profile() {
    hardware_profile_ = detect_hardware();
}

OffloadDecision HardwareAbstractionLayer::select_optimal_backend(const ComputationTask& task) {
    if (current_backend_ != ComputeBackend::AUTO) {
        return OffloadDecision{
            current_backend_,
            OffloadReason::PERFORMANCE_OPTIMIZATION,
            0.9,
            "Using forced backend: " + std::to_string(static_cast<int>(current_backend_))
        };
    }

    return make_offload_decision(task);
}

ComputeBackend HardwareAbstractionLayer::get_current_backend() {
    return current_backend_;
}

void HardwareAbstractionLayer::force_backend(ComputeBackend backend) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_backend_ = backend;
    std::cout << "Forced backend to: " << static_cast<int>(backend) << std::endl;
}

void HardwareAbstractionLayer::execute_task(const ComputationTask& task) {
    OffloadDecision decision = select_optimal_backend(task);

    switch (decision.selected_backend) {
        case ComputeBackend::GPU_VULKAN:
            if (is_gpu_available()) {
                execute_on_gpu(task);
            } else {
                std::cout << "GPU requested but not available, falling back to CPU" << std::endl;
                execute_on_cpu(task);
            }
            break;

        case ComputeBackend::CPU_PARALLEL:
        case ComputeBackend::CPU_SEQUENTIAL:
            execute_on_cpu(task);
            break;

        default:
            execute_on_cpu(task);
            break;
    }
}

void HardwareAbstractionLayer::execute_task_async(const ComputationTask& task,
                                                 std::function<void(const ComputationTask&)> callback) {
    std::thread([this, task, callback]() {
        execute_task(task);
        if (callback) {
            callback(task);
        }
    }).detach();
}

void HardwareAbstractionLayer::queue_task(const ComputationTask& task) {
    std::lock_guard<std::mutex> lock(mutex_);
    task_queue_.push_back(task);
    task_available_cv_.notify_one();
}

void HardwareAbstractionLayer::process_task_queue() {
    if (!running_) {
        running_ = true;
        task_processor_thread_ = std::thread(&HardwareAbstractionLayer::task_processor_loop, this);
        health_monitor_thread_ = std::thread(&HardwareAbstractionLayer::health_monitor_loop, this);
    }
}

size_t HardwareAbstractionLayer::get_queued_task_count() {
    std::lock_guard<std::mutex> lock(mutex_);
    return task_queue_.size();
}

std::unordered_map<std::string, double> HardwareAbstractionLayer::get_backend_performance_stats() {
    std::unordered_map<std::string, double> stats;

    stats["gpu_utilization"] = get_gpu_utilization();
    stats["cpu_utilization"] = get_cpu_utilization();
    stats["active_gpu_tasks"] = active_tasks_per_backend_[ComputeBackend::GPU_VULKAN].load();
    stats["active_cpu_tasks"] = active_tasks_per_backend_[ComputeBackend::CPU_PARALLEL].load() +
                               active_tasks_per_backend_[ComputeBackend::CPU_SEQUENTIAL].load();

    return stats;
}

double HardwareAbstractionLayer::get_gpu_utilization() {
    // Simplified GPU utilization check
    if (!is_gpu_available()) return 0.0;

    // In a real implementation, this would query GPU driver APIs
    return active_tasks_per_backend_[ComputeBackend::GPU_VULKAN].load() > 0 ? 0.8 : 0.1;
}

double HardwareAbstractionLayer::get_cpu_utilization() {
    if (!resource_monitor_) return 0.0;

    auto usage = resource_monitor_->get_current_usage();
    return usage.cpu_usage_percent / 100.0;
}

bool HardwareAbstractionLayer::switch_to_gpu() {
    if (is_gpu_available()) {
        force_backend(ComputeBackend::GPU_VULKAN);
        return initialize_vulkan_backend();
    }
    return false;
}

bool HardwareAbstractionLayer::switch_to_cpu() {
    force_backend(ComputeBackend::CPU_PARALLEL);
    return initialize_cpu_backend();
}

bool HardwareAbstractionLayer::switch_to_auto() {
    force_backend(ComputeBackend::AUTO);
    return true;
}

bool HardwareAbstractionLayer::is_backend_healthy(ComputeBackend backend) {
    switch (backend) {
        case ComputeBackend::GPU_VULKAN:
            return is_gpu_available() && hardware_profile_.vulkan_supported;
        case ComputeBackend::CPU_PARALLEL:
        case ComputeBackend::CPU_SEQUENTIAL:
            return hardware_profile_.cpu_cores > 0;
        default:
            return true;
    }
}

std::vector<std::string> HardwareAbstractionLayer::get_backend_warnings() {
    std::vector<std::string> warnings;

    if (!is_gpu_available()) {
        warnings.push_back("GPU not available, falling back to CPU");
    }

    if (hardware_profile_.system_memory_mb < 4096) {
        warnings.push_back("Low system memory may impact performance");
    }

    if (active_tasks_per_backend_[ComputeBackend::GPU_VULKAN].load() > 10) {
        warnings.push_back("High GPU task load may cause performance degradation");
    }

    return warnings;
}

void HardwareAbstractionLayer::recover_from_backend_failure(ComputeBackend backend) {
    std::cout << "Attempting recovery from backend failure: " << static_cast<int>(backend) << std::endl;

    switch (backend) {
        case ComputeBackend::GPU_VULKAN:
            handle_gpu_failure();
            break;
        case ComputeBackend::CPU_PARALLEL:
        case ComputeBackend::CPU_SEQUENTIAL:
            handle_cpu_failure();
            break;
        default:
            break;
    }
}

// Private methods

HardwareProfile HardwareAbstractionLayer::detect_gpu_capabilities() {
    HardwareProfile profile;

    // Universal CPU core detection
#ifdef _WIN32
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    profile.cpu_cores = sysInfo.dwNumberOfProcessors;
#elif __linux__
    profile.cpu_cores = sysconf(_SC_NPROCESSORS_ONLN);
#else
    profile.cpu_cores = 1; // Fallback
#endif

    // Universal system memory detection
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    profile.system_memory_mb = memInfo.ullTotalPhys / (1024 * 1024);
#elif __linux__
    struct sysinfo memInfo;
    sysinfo(&memInfo);
    profile.system_memory_mb = (memInfo.totalram * memInfo.mem_unit) / (1024 * 1024);
#else
    profile.system_memory_mb = 1024; // Fallback 1GB
#endif

    // Try to detect Vulkan-capable GPU
    VkInstance instance;
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Nyx Hardware Detection";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
    if (result == VK_SUCCESS) {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount > 0) {
            std::vector<VkPhysicalDevice> devices(deviceCount);
            VkResult enumResult = vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

            if (enumResult == VK_SUCCESS && deviceCount > 0 && devices[0] != VK_NULL_HANDLE) {
                VkPhysicalDeviceProperties deviceProperties;
                vkGetPhysicalDeviceProperties(devices[0], &deviceProperties);

                profile.gpu_name = deviceProperties.deviceName;
                profile.vulkan_supported = true;

                VkPhysicalDeviceMemoryProperties memProperties;
                vkGetPhysicalDeviceMemoryProperties(devices[0], &memProperties);

                uint64_t totalGpuMemory = 0;
                for (uint32_t i = 0; i < memProperties.memoryHeapCount; i++) {
                    if (memProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                        totalGpuMemory += memProperties.memoryHeaps[i].size;
                    }
                }
                profile.gpu_memory_mb = totalGpuMemory / (1024 * 1024);
                profile.gpu_compute_units = deviceProperties.limits.maxComputeWorkGroupCount[0];

                vkDestroyInstance(instance, nullptr);
            } else {
                // Enumeration failed or no valid devices
                profile.vulkan_supported = false;
                profile.gpu_memory_mb = 0;
                profile.gpu_compute_units = 0;
                profile.gpu_name = "Vulkan device enumeration failed";
                vkDestroyInstance(instance, nullptr);
            }
    } else {
        profile.vulkan_supported = false;
        profile.gpu_memory_mb = 0;
        profile.gpu_compute_units = 0;
        profile.gpu_name = "No Vulkan GPU";
        std::cout << "Vulkan not supported, GPU acceleration unavailable" << std::endl;
    }
}

    // Check for CUDA (simplified)
    profile.cuda_supported = false; // Would need CUDA runtime check

    return profile;
}

HardwareProfile HardwareAbstractionLayer::detect_cpu_capabilities() {
    HardwareProfile profile;

    // Read CPU info from /proc/cpuinfo
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    int cores = 0;
    std::string cpu_name;

    while (std::getline(cpuinfo, line)) {
        if (line.find("processor") != std::string::npos) {
            cores++;
        }
        if (line.find("model name") != std::string::npos) {
            size_t colon_pos = line.find(':');
            if (colon_pos != std::string::npos) {
                cpu_name = line.substr(colon_pos + 2);
            }
        }
    }

    profile.cpu_cores = cores > 0 ? cores : std::thread::hardware_concurrency();
    profile.cpu_name = cpu_name.empty() ? "Unknown CPU" : cpu_name;

    return profile;
}

HardwareProfile HardwareAbstractionLayer::detect_system_capabilities() {
    HardwareProfile profile;

    // Get system memory info
#ifdef _WIN32
    MEMORYSTATUSEX memStatus;
    memStatus.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memStatus)) {
        profile.system_memory_mb = memStatus.ullTotalPhys / (1024 * 1024);
    }
    // Get disk info
    ULARGE_INTEGER freeBytesAvailable, totalNumberOfBytes, totalNumberOfFreeBytes;
    if (GetDiskFreeSpaceExA("C:\\", &freeBytesAvailable, &totalNumberOfBytes, &totalNumberOfFreeBytes)) {
        profile.available_storage_mb = freeBytesAvailable.QuadPart / (1024 * 1024);
    }
#else
    struct sysinfo sys_info;
    if (sysinfo(&sys_info) == 0) {
        profile.system_memory_mb = sys_info.totalram / (1024 * 1024);
    }

    // Get disk info
    struct statvfs stat;
    if (statvfs("/", &stat) == 0) {
        profile.available_storage_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024);
    }
#endif

    return profile;
}

bool HardwareAbstractionLayer::test_vulkan_support() {
    VkInstance instance;
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
    if (result == VK_SUCCESS) {
        vkDestroyInstance(instance, nullptr);
        return true;
    }
    return false;
}

bool HardwareAbstractionLayer::test_cuda_support() {
    // Simplified CUDA check - would need CUDA runtime
    return false;
}

bool HardwareAbstractionLayer::initialize_vulkan_backend() {
    // Vulkan backend initialization would go here
    // For now, just check if Vulkan is supported
    return is_vulkan_supported();
}

bool HardwareAbstractionLayer::initialize_cpu_backend() {
    // CPU backend is always available
    return hardware_profile_.cpu_cores > 0;
}

void HardwareAbstractionLayer::cleanup_backends() {
    // Cleanup backend resources
    std::cout << "Cleaning up hardware backends" << std::endl;
}

void HardwareAbstractionLayer::task_processor_loop() {
    while (running_) {
        ComputationTask task;

        {
            std::unique_lock<std::mutex> lock(mutex_);
            task_available_cv_.wait_for(lock, std::chrono::milliseconds(100), [this]() {
                return !task_queue_.empty() || !running_;
            });

            if (!running_) break;

            if (!task_queue_.empty()) {
                task = task_queue_.front();
                task_queue_.erase(task_queue_.begin());
            } else {
                continue;
            }
        }

        // Execute the task
        execute_task(task);
    }
}

void HardwareAbstractionLayer::health_monitor_loop() {
    while (running_) {
        // Monitor backend health
        for (auto backend : {ComputeBackend::GPU_VULKAN, ComputeBackend::CPU_PARALLEL}) {
            if (!is_backend_healthy(backend)) {
                recover_from_backend_failure(backend);
            }
        }

        // Update resource usage stats
        update_resource_usage_stats();

        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}

void HardwareAbstractionLayer::execute_on_gpu(const ComputationTask& task) {
    active_tasks_per_backend_[ComputeBackend::GPU_VULKAN]++;
    try {
        if (task.gpu_task) {
            task.gpu_task();
        } else {
            std::cout << "No GPU task implementation for: " << task.description << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "GPU task failed: " << e.what() << std::endl;
    }
    active_tasks_per_backend_[ComputeBackend::GPU_VULKAN]--;
}

void HardwareAbstractionLayer::execute_on_cpu(const ComputationTask& task) {
    ComputeBackend backend = (hardware_profile_.cpu_cores >= 4) ?
                            ComputeBackend::CPU_PARALLEL : ComputeBackend::CPU_SEQUENTIAL;

    active_tasks_per_backend_[backend]++;
    try {
        if (task.cpu_task) {
            task.cpu_task();
        } else {
            std::cout << "No CPU task implementation for: " << task.description << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "CPU task failed: " << e.what() << std::endl;
    }
    active_tasks_per_backend_[backend]--;
}

OffloadDecision HardwareAbstractionLayer::analyze_task_requirements(const ComputationTask& task) {
    OffloadDecision decision;
    decision.selected_backend = ComputeBackend::CPU_PARALLEL;
    decision.confidence_score = 0.5;

    // Analyze task requirements and select best backend
    if (task.preferred_backend != ComputeBackend::AUTO) {
        decision.selected_backend = task.preferred_backend;
        decision.reason = OffloadReason::PERFORMANCE_OPTIMIZATION;
        decision.confidence_score = 0.8;
        decision.explanation = "Using preferred backend";
        return decision;
    }

    // GPU is preferred for compute-intensive tasks
    if (task.estimated_gpu_memory_mb > 0 && is_gpu_available()) {
        if (task.estimated_gpu_memory_mb < hardware_profile_.gpu_memory_mb * 0.8) {
            decision.selected_backend = ComputeBackend::GPU_VULKAN;
            decision.reason = OffloadReason::PERFORMANCE_OPTIMIZATION;
            decision.confidence_score = 0.9;
            decision.explanation = "GPU has sufficient memory and is available";
        } else {
            decision.selected_backend = ComputeBackend::CPU_PARALLEL;
            decision.reason = OffloadReason::GPU_MEMORY_LOW;
            decision.confidence_score = 0.7;
            decision.explanation = "Task requires more GPU memory than available";
        }
    } else if (!is_gpu_available()) {
        decision.selected_backend = ComputeBackend::CPU_PARALLEL;
        decision.reason = OffloadReason::GPU_UNAVAILABLE;
        decision.confidence_score = 0.6;
        decision.explanation = "GPU not available, using CPU";
    }

    return decision;
}

OffloadDecision HardwareAbstractionLayer::make_offload_decision(const ComputationTask& task) {
    OffloadDecision decision = analyze_task_requirements(task);

    // Check current resource usage
    if (!check_resource_availability(decision.selected_backend, task)) {
        // Fallback to CPU if resources not available
        decision.selected_backend = ComputeBackend::CPU_PARALLEL;
        decision.reason = OffloadReason::GPU_OVERLOADED;
        decision.confidence_score = 0.4;
        decision.explanation = "Target backend overloaded, falling back to CPU";
    }

    return decision;
}

double HardwareAbstractionLayer::calculate_backend_score(ComputeBackend backend, const ComputationTask& task) {
    double score = 0.0;

    switch (backend) {
        case ComputeBackend::GPU_VULKAN:
            if (is_gpu_available()) {
                score = 1.0 - (active_tasks_per_backend_[backend].load() * 0.1);
                score *= (task.estimated_gpu_memory_mb < hardware_profile_.gpu_memory_mb * 0.8) ? 1.0 : 0.3;
            }
            break;

        case ComputeBackend::CPU_PARALLEL:
            score = 1.0 - (active_tasks_per_backend_[backend].load() * 0.05);
            score *= (hardware_profile_.cpu_cores >= 4) ? 1.0 : 0.7;
            break;

        case ComputeBackend::CPU_SEQUENTIAL:
            score = 0.5; // Always available but slower
            break;

        default:
            score = 0.1;
            break;
    }

    return std::max<double>(0.0, std::min<double>(1.0, score));
}

bool HardwareAbstractionLayer::check_resource_availability(ComputeBackend backend, const ComputationTask& task) {
    if (!resource_monitor_) return true;

    auto usage = resource_monitor_->get_current_usage();

    switch (backend) {
        case ComputeBackend::GPU_VULKAN:
            return usage.memory_usage_percent < 80.0; // Simplified check

        case ComputeBackend::CPU_PARALLEL:
        case ComputeBackend::CPU_SEQUENTIAL:
            return usage.cpu_usage_percent < 85.0 && usage.memory_usage_percent < 90.0;

        default:
            return true;
    }
}

void HardwareAbstractionLayer::update_resource_usage_stats() {
    // Update internal resource tracking
    // This would integrate with the resource monitor
}

void HardwareAbstractionLayer::handle_gpu_failure() {
    std::cout << "Handling GPU failure - switching to CPU" << std::endl;
    force_backend(ComputeBackend::CPU_PARALLEL);
}

void HardwareAbstractionLayer::handle_cpu_failure() {
    std::cout << "CPU failure detected - attempting recovery" << std::endl;
    // In a real implementation, this would try to restart CPU operations
}

void HardwareAbstractionLayer::log_backend_switch(ComputeBackend from, ComputeBackend to, OffloadReason reason) {
    std::cout << "Backend switch: " << static_cast<int>(from) << " -> " << static_cast<int>(to)
              << " (reason: " << static_cast<int>(reason) << ")" << std::endl;
}