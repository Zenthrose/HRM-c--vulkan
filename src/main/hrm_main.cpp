#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <vulkan/vulkan.h>

#include "../vulkan/vulkan_compatibility.h"
#include "../hrm/resource_aware_hrm.hpp"
#include "../system/memory_compaction_system.hpp"
#include "../system/cloud_storage_manager.hpp"
#include "hrm_cli.hpp"
#include "hrm_gui.hpp"
#include "../training/character_language_trainer.hpp"
#include "../system/hardware_profiler.hpp"
#include "../utils/logger.hpp"
#include "../utils/config_manager.hpp"

namespace fs = std::filesystem;

// Default configuration with hardware adaptation
ResourceAwareHRMConfig createDefaultHRMConfig(const HardwareCapabilities& hw_caps) {
    ResourceAwareHRMConfig config;

    // Base HRM config (SelfModifyingHRMConfig)
    config.base_config.base_config.use_utf8_communication = true;
    config.base_config.base_config.max_conversation_length = 10000;
    config.base_config.base_config.enable_self_evolution = true;
    config.base_config.base_config.evolution_rate = 0.1f;
    config.base_config.base_config.adaptation_cycles = 10;
    config.base_config.base_config.enable_continual_learning = true;

    // UTF-8 configuration
    config.base_config.base_config.utf8_config.max_sequence_length = 2048;
    config.base_config.base_config.utf8_config.embedding_dim = 256;
    config.base_config.base_config.utf8_config.use_byte_fallback = true;

    // Use environment variable for project root, default to system root for full scanning
    if (const char* env_root = std::getenv("HRM_PROJECT_ROOT")) {
        config.base_config.project_root = env_root;
    } else {
        // Default to system root for comprehensive scanning capability
        config.base_config.project_root = "C:/"; // Windows system root
    }

    // Use temp directory for compilation
    config.base_config.temp_compilation_dir = (fs::temp_directory_path() / "hrm_temp").string();
    config.base_config.enable_self_modification = true; // Enabled for autonomous learning
    config.base_config.enable_runtime_recompilation = true;
    config.base_config.self_analysis_frequency = 0.05f;
    config.base_config.modification_confidence_threshold = 0.95f;
    config.base_config.create_backups_before_modification = true;

    // Resource monitoring
    config.enable_resource_monitoring = true;
    config.enable_adaptive_task_management = true;
    config.enable_chunking_for_large_tasks = true;
    config.resource_check_interval = std::chrono::seconds(60);
    config.max_memory_per_task_mb = 512;
    config.max_cpu_per_task_percent = 80.0;

    return config;
}

struct VulkanResources {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    uint32_t computeQueueFamilyIndex = 0;
    VkCommandPool commandPool = VK_NULL_HANDLE;
};

void adapt_config_to_hardware(ResourceAwareHRMConfig& config, const HardwareCapabilities& hw_caps, const VulkanResources& vulkan) {
    uint64_t ram_gb = hw_caps.total_ram_bytes / (1024ULL * 1024 * 1024);
    uint32_t cores = hw_caps.cpu_cores;

    std::cout << "Adapting HRM configuration for hardware tier: ";
    switch (hw_caps.performance_tier) {
        case HardwareCapabilities::PerformanceTier::ULTRA_LOW:
            std::cout << "Ultra Low" << std::endl;
            // Minimal configuration for embedded systems
            config.base_config.base_config.hrm_config.inner_config.hidden_size = 64;
            config.base_config.base_config.hrm_config.inner_config.H_layers = 1;
            config.base_config.base_config.hrm_config.inner_config.L_layers = 1;
            config.base_config.base_config.hrm_config.inner_config.num_heads = 1;
            config.base_config.base_config.hrm_config.inner_config.vocab_size = 10000;
            config.base_config.base_config.hrm_config.inner_config.batch_size = 1;
            config.base_config.base_config.hrm_config.inner_config.seq_len = 128;
            config.max_memory_per_task_mb = 50;
            config.base_config.enable_self_modification = false; // Too risky for minimal hardware
            config.enable_adaptive_task_management = false;
            break;

        case HardwareCapabilities::PerformanceTier::LOW:
            std::cout << "Low" << std::endl;
            // Reduced configuration for low-end systems
            config.base_config.base_config.hrm_config.inner_config.hidden_size = 128;
            config.base_config.base_config.hrm_config.inner_config.H_layers = 2;
            config.base_config.base_config.hrm_config.inner_config.L_layers = 2;
            config.base_config.base_config.hrm_config.inner_config.num_heads = 2;
            config.base_config.base_config.hrm_config.inner_config.vocab_size = 25000;
            config.base_config.base_config.hrm_config.inner_config.batch_size = 1;
            config.base_config.base_config.hrm_config.inner_config.seq_len = 256;
            config.max_memory_per_task_mb = 100;
            config.base_config.enable_self_modification = false; // Conservative
            break;

        case HardwareCapabilities::PerformanceTier::MEDIUM:
            std::cout << "Medium" << std::endl;
            // Balanced configuration
            config.base_config.base_config.hrm_config.inner_config.hidden_size = 256;
            config.base_config.base_config.hrm_config.inner_config.H_layers = 3;
            config.base_config.base_config.hrm_config.inner_config.L_layers = 3;
            config.base_config.base_config.hrm_config.inner_config.num_heads = 4;
            config.base_config.base_config.hrm_config.inner_config.vocab_size = 50000;
            config.base_config.base_config.hrm_config.inner_config.batch_size = 2;
            config.base_config.base_config.hrm_config.inner_config.seq_len = 512;
            config.max_memory_per_task_mb = 256;
            config.base_config.enable_self_modification = true; // Enable with caution
            break;

        case HardwareCapabilities::PerformanceTier::HIGH:
            std::cout << "High" << std::endl;
            // Good performance configuration
            config.base_config.base_config.hrm_config.inner_config.hidden_size = 512;
            config.base_config.base_config.hrm_config.inner_config.H_layers = 4;
            config.base_config.base_config.hrm_config.inner_config.L_layers = 4;
            config.base_config.base_config.hrm_config.inner_config.num_heads = 8;
            config.base_config.base_config.hrm_config.inner_config.vocab_size = 75000;
            config.base_config.base_config.hrm_config.inner_config.batch_size = 4;
            config.base_config.base_config.hrm_config.inner_config.seq_len = 1024;
            config.max_memory_per_task_mb = 512;
            config.base_config.enable_self_modification = true;
            break;

        case HardwareCapabilities::PerformanceTier::ULTRA_HIGH:
            std::cout << "Ultra High" << std::endl;
            // Full capability configuration (original)
            config.base_config.base_config.hrm_config.inner_config.hidden_size = 768;
            config.base_config.base_config.hrm_config.inner_config.H_layers = 4;
            config.base_config.base_config.hrm_config.inner_config.L_layers = 4;
            config.base_config.base_config.hrm_config.inner_config.num_heads = 12;
            config.base_config.base_config.hrm_config.inner_config.vocab_size = 100000;
            config.base_config.base_config.hrm_config.inner_config.batch_size = 8;
            config.base_config.base_config.hrm_config.inner_config.seq_len = 2048;
            config.max_memory_per_task_mb = 1024;
            config.base_config.enable_self_modification = true;
            break;
    }

    // Vulkan availability adjustments
    if (!hw_caps.vulkan_supported || vulkan.device == VK_NULL_HANDLE) {
        std::cout << "Vulkan not available - enabling CPU-only mode" << std::endl;
        // CPU-only mode - reduce complexity further
        config.base_config.base_config.hrm_config.inner_config.hidden_size = std::min(config.base_config.base_config.hrm_config.inner_config.hidden_size, 256);
        config.base_config.base_config.hrm_config.inner_config.H_layers = std::min(config.base_config.base_config.hrm_config.inner_config.H_layers, 2);
        config.base_config.base_config.hrm_config.inner_config.L_layers = std::min(config.base_config.base_config.hrm_config.inner_config.L_layers, 2);
        config.base_config.enable_self_modification = false; // Too complex without GPU
    }

    // Resource monitoring adjustments
    if (ram_gb < 2) {
        config.enable_resource_monitoring = true;
        config.resource_check_interval = std::chrono::seconds(30); // More frequent checks
    } else if (ram_gb < 8) {
        config.enable_resource_monitoring = true;
        config.resource_check_interval = std::chrono::seconds(60);
    } else {
        config.enable_resource_monitoring = true;
        config.resource_check_interval = std::chrono::seconds(120);
    }

    // Task management based on CPU cores
    if (cores <= 1) {
        config.enable_adaptive_task_management = false;
        config.enable_chunking_for_large_tasks = false;
    } else if (cores <= 2) {
        config.enable_adaptive_task_management = true;
        config.enable_chunking_for_large_tasks = true;
    } else {
        config.enable_adaptive_task_management = true;
        config.enable_chunking_for_large_tasks = true;
    }

    std::cout << "Configuration adapted - Hidden size: " << config.base_config.base_config.hrm_config.inner_config.hidden_size
              << ", Layers: " << config.base_config.base_config.hrm_config.inner_config.H_layers
              << ", Self-modification: " << (config.base_config.enable_self_modification ? "Enabled" : "Disabled") << std::endl;
}

VulkanResources initializeVulkan() {
    VulkanResources res;

    // Check Vulkan compatibility
    VulkanCompatibilityInfo compat = VulkanCompatibility::checkCompatibility();

    if (!compat.vulkanSupported) {
        throw std::runtime_error("Vulkan is not supported on this system");
    }

    if (compat.devices.empty()) {
        throw std::runtime_error("No Vulkan-compatible devices found");
    }

    if (!compat.selectedDevice) {
        throw std::runtime_error("No suitable Vulkan device selected");
    }

    std::cout << "Selected Vulkan device: " << compat.selectedDevice->name << std::endl;
    std::cout << "Device score: " << compat.selectedDevice->score << std::endl;

    // Create Vulkan instance with negotiated API version
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "HRM System";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "HRM Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = compat.apiVersion;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    // Disable validation layers for now to avoid buffer usage flag issues
    std::vector<const char*> layers;

    createInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
    createInfo.ppEnabledLayerNames = layers.data();

    if (vkCreateInstance(&createInfo, nullptr, &res.instance) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance");
    }

    // Use selected device
    res.physicalDevice = compat.selectedDevice->device;

    // Use compute queue family from compatibility check
    if (compat.selectedDevice->computeQueueFamily == UINT32_MAX) {
        throw std::runtime_error("Selected device does not have a compute queue family");
    }
    res.computeQueueFamilyIndex = compat.selectedDevice->computeQueueFamily;

    // Check device features
    if (!VulkanCompatibility::checkDeviceFeatures(res.physicalDevice)) {
        std::cerr << "Warning: Selected device may not support all required features" << std::endl;
    }

    // Create logical device
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = res.computeQueueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    float queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCI{};
    deviceCI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCI.queueCreateInfoCount = 1;
    deviceCI.pQueueCreateInfos = &queueCreateInfo;
    deviceCI.enabledLayerCount = static_cast<uint32_t>(layers.size());
    deviceCI.ppEnabledLayerNames = layers.data();

    // Enable required features
    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.shaderInt64 = VK_TRUE; // For some operations
    deviceCI.pEnabledFeatures = &deviceFeatures;

    VkResult deviceResult = vkCreateDevice(res.physicalDevice, &deviceCI, nullptr, &res.device);
    if (deviceResult != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan device: " +
                                VulkanCompatibility::getVulkanResultString(deviceResult));
    }

    // Get compute queue
    vkGetDeviceQueue(res.device, res.computeQueueFamilyIndex, 0, &res.computeQueue);

    // Create command pool
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = res.computeQueueFamilyIndex;

    if (vkCreateCommandPool(res.device, &poolInfo, nullptr, &res.commandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }

    return res;
}

MemoryCompactionConfig createDefaultMemoryConfig(std::shared_ptr<CloudStorageManager> cloud_manager, ConfigManager& config_manager) {
    MemoryCompactionConfig config;

    // Load settings from ConfigManager
    auto& logger = Logger::getInstance();

    // Set defaults, override with config values
    config.default_level = MemoryCompactionLevel::MEDIUM;
    std::string level = config_manager.get_string("memory_compaction.default_level", "MEDIUM");
    if (level == "LIGHT") config.default_level = MemoryCompactionLevel::LIGHT;
    else if (level == "HEAVY") config.default_level = MemoryCompactionLevel::HEAVY;
    else if (level == "EXTREME") config.default_level = MemoryCompactionLevel::EXTREME;

    config.preferred_algorithm = CompressionAlgorithm::LZ4;
    std::string algo = config_manager.get_string("memory_compaction.preferred_algorithm", "LZ4");
    if (algo == "ZSTD") config.preferred_algorithm = CompressionAlgorithm::ZSTD;
    else if (algo == "GZIP") config.preferred_algorithm = CompressionAlgorithm::GZIP;
    else if (algo == "BROTLI") config.preferred_algorithm = CompressionAlgorithm::BROTLI;
    else if (algo == "NONE") config.preferred_algorithm = CompressionAlgorithm::NONE;

    config.max_memory_before_compaction_mb = config_manager.get_int("memory_compaction.max_memory_before_compaction_mb", 1024);
    config.target_memory_after_compaction_mb = config_manager.get_int("memory_compaction.target_memory_after_compaction_mb", 512);
    config.auto_compaction_enabled = config_manager.get_bool("memory_compaction.auto_compaction_enabled", true);

    int interval_hours = config_manager.get_int("memory_compaction.compaction_interval_hours", 6);
    config.compaction_interval = std::chrono::hours(interval_hours);

    config.preserve_recent_conversations = config_manager.get_bool("memory_compaction.preserve_recent_conversations", true);

    int window_hours = config_manager.get_int("memory_compaction.recent_conversation_window_hours", 24);
    config.recent_conversation_window = std::chrono::hours(window_hours);

    // Use environment variable or temp directory for compaction
    if (const char* env_dir = std::getenv("HRM_COMPACTION_DIR")) {
        config.compaction_directory = env_dir;
    } else {
        config.compaction_directory = (fs::temp_directory_path() / "hrm_compactions").string();
    }
    logger.info("Memory compaction directory: " + config.compaction_directory);

    config.cloud_storage_manager = cloud_manager;
    config.default_cloud_provider = CloudProvider::LOCAL_STORAGE;
    std::string provider = config_manager.get_string("general.default_cloud_provider");
    if (!provider.empty()) {
        if (provider == "GOOGLE_DRIVE") config.default_cloud_provider = CloudProvider::GOOGLE_DRIVE;
        else if (provider == "DROPBOX") config.default_cloud_provider = CloudProvider::DROPBOX;
        else if (provider == "ONEDRIVE") config.default_cloud_provider = CloudProvider::ONEDRIVE;
        else if (provider == "MEGA") config.default_cloud_provider = CloudProvider::MEGA;
    }

    return config;
}

void setupCloudStorage(CloudStorageManager& cloud_manager) {
    // Read config file for cloud provider settings
    std::ifstream config_file("./config/hrm_config.txt");
    std::unordered_map<std::string, std::string> settings;

    if (config_file.is_open()) {
        std::string line;
        std::string current_section;
        while (std::getline(config_file, line)) {
            // Remove comments and trim
            size_t comment_pos = line.find('#');
            if (comment_pos != std::string::npos) {
                line = line.substr(0, comment_pos);
            }
            line.erase(line.begin(), std::find_if(line.begin(), line.end(), [](int ch) { return !std::isspace(ch); }));
            line.erase(std::find_if(line.rbegin(), line.rend(), [](int ch) { return !std::isspace(ch); }).base(), line.end());

            if (line.empty()) continue;

            if (line[0] == '[' && line.back() == ']') {
                current_section = line.substr(1, line.size() - 2);
            } else if (!current_section.empty()) {
                size_t equals_pos = line.find('=');
                if (equals_pos != std::string::npos) {
                    std::string key = current_section + "." + line.substr(0, equals_pos);
                    std::string value = line.substr(equals_pos + 1);
                    key.erase(key.begin(), std::find_if(key.begin(), key.end(), [](int ch) { return !std::isspace(ch); }));
                    key.erase(std::find_if(key.rbegin(), key.rend(), [](int ch) { return !std::isspace(ch); }).base(), key.end());
                    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](int ch) { return !std::isspace(ch); }));
                    value.erase(std::find_if(value.rbegin(), value.rend(), [](int ch) { return !std::isspace(ch); }).base(), value.end());
                    settings[key] = value;
                }
            }
        }
    }

    // Add local storage provider (always available)
    CloudStorageConfig local_config;
    local_config.provider = CloudProvider::LOCAL_STORAGE;
    local_config.compaction_directory = settings.count("general.cloud_storage_directory") ?
        settings["general.cloud_storage_directory"] : "./hrm_cloud_storage";

    auto local_provider = std::make_shared<LocalStorageProvider>(local_config);
    cloud_manager.add_provider(local_provider);

    // Set default provider
    CloudProvider default_provider = CloudProvider::LOCAL_STORAGE;
    if (settings.count("general.default_cloud_provider")) {
        std::string provider = settings["general.default_cloud_provider"];
        if (provider == "GOOGLE_DRIVE") default_provider = CloudProvider::GOOGLE_DRIVE;
        else if (provider == "DROPBOX") default_provider = CloudProvider::DROPBOX;
        else if (provider == "ONEDRIVE") default_provider = CloudProvider::ONEDRIVE;
        else if (provider == "MEGA") default_provider = CloudProvider::MEGA;
    }
    cloud_manager.set_default_provider(default_provider);

    std::cout << "Cloud storage initialized with local storage provider" << std::endl;
}

void runCLI(std::shared_ptr<ResourceAwareHRM> hrm) {
    HRMCLI cli(hrm);
    cli.run();
}

void runGUI(std::shared_ptr<ResourceAwareHRM> hrm) {
    HRMGUI gui(hrm);
    gui.run();
}

void runCharacterTraining(std::shared_ptr<ResourceAwareHRM> hrm) {
    std::cout << "Starting Character-Level Language Training Mode" << std::endl;
    std::cout << "==================================================" << std::endl;

    // Create character language training configuration
    CharacterLanguageModelConfig train_config;
    train_config.max_epochs = 10;  // Shorter for demo
    train_config.batch_size = 2;   // Smaller batch size
    train_config.max_seq_length = 512;  // Shorter sequences
    train_config.context_length = 256;
    train_config.learning_rate = 1e-4;
    train_config.warmup_steps = 100;
    train_config.total_steps = 10000;

    // Initialize character language trainer
    CharacterLanguageTrainer trainer(hrm, train_config);

    // Run training
    std::string dataset_path = "./data/text/processed";
    std::unordered_map<std::string, float> training_results;
    try {
        std::cout << "Starting training with dataset path: " << dataset_path << std::endl;
        training_results = trainer.train_character_language_model(dataset_path);
        std::cout << "Training completed successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Training failed with exception: " << e.what() << std::endl;
        std::cerr << "This may be causing the task scheduler to stop unexpectedly" << std::endl;
        return; // Exit early to prevent destructor issues
    }

    std::cout << "\n Character-level training completed!" << std::endl;
    std::cout << "Final Results:" << std::endl;
    for (const auto& [metric, value] : training_results) {
        std::cout << "  " << metric << ": " << value << std::endl;
    }

    // Test text generation
    std::cout << "\nTesting text generation..." << std::endl;
    std::string prompt = "The quick brown fox";
    std::string generated = trainer.generate_text(prompt, 100);
    std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
    std::cout << "Generated: \"" << generated << "\"" << std::endl;
}

void printUsage(const char* program_name) {
    std::cout << "HRM - Hierarchical Reasoning Module" << std::endl;
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --cli              Run in command-line interface mode" << std::endl;
    std::cout << "  --gui              Run in graphical user interface mode" << std::endl;
    std::cout << "  --train            Run character-level language training" << std::endl;
    std::cout << "  --test             Run basic functionality tests" << std::endl;
    std::cout << "  --help             Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "If no options are provided, GUI mode is started by default." << std::endl;
}

void runBasicTests(std::shared_ptr<ResourceAwareHRM> hrm,
                   std::shared_ptr<MemoryCompactionSystem> memory_system,
                   std::shared_ptr<CloudStorageManager> cloud_manager) {
    std::cout << "Running basic functionality tests..." << std::endl;

    // Test memory compaction (without HRM to avoid Vulkan issues)
    std::cout << "Testing memory compaction..." << std::endl;
    std::vector<ConversationEntry> test_entries = {
        {"test1", std::chrono::system_clock::now(), "User: Hello", "HRM: Hi there!", 0.95, {"greeting"}, {"user"}, 100},
        {"test2", std::chrono::system_clock::now(), "User: How are you?", "HRM: I'm doing well, thank you!", 0.92, {"conversation"}, {"user"}, 120}
    };

    auto compaction_result = memory_system->compact_memory(test_entries);
    if (compaction_result.success) {
        std::cout << "Memory compaction successful. ID: " << compaction_result.compaction_id << std::endl;

        // Test decompression
        auto decompressed = memory_system->decompress_memory(compaction_result.compaction_id);
        std::cout << "Decompressed " << decompressed.size() << " conversation entries" << std::endl;
    } else {
        std::cout << "Memory compaction failed: " << compaction_result.error_message << std::endl;
    }

    // Test cloud storage
    std::cout << "Testing cloud storage..." << std::endl;
    auto providers = cloud_manager->get_available_providers();
    std::cout << "Available cloud providers: " << providers.size() << std::endl;

    // Test memory statistics
    auto mem_stats = memory_system->get_memory_stats();
    std::cout << "Memory stats - Total compactions: " << mem_stats["total_compactions"] << std::endl;

    std::cout << "Basic tests completed successfully!" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "HRM - Hierarchical Reasoning Module" << std::endl;
    std::cout << "===================================" << std::endl;

    // Hardware profiling for universal compatibility
    HardwareProfiler hw_profiler;
    HardwareCapabilities hw_caps = hw_profiler.profile_system();

    // Initialize logging system
    auto& logger = Logger::getInstance();
    logger.info("HRM System starting up");

    // Initialize configuration manager
    ConfigManager config_manager;
    logger.info("Configuration loaded from: " + config_manager.getConfigDir());

    // Parse command line arguments
    bool cli_mode = false;
    bool gui_mode = false;
    bool train_mode = false;
    bool test_mode = false;
    bool show_help = false;

    bool invalid_arg = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cli") {
            cli_mode = true;
        } else if (arg == "--gui") {
            gui_mode = true;
        } else if (arg == "--train") {
            train_mode = true;
        } else if (arg == "--test") {
            test_mode = true;
        } else if (arg == "--help") {
            show_help = true;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            invalid_arg = true;
            show_help = true;
            break;
        }
    }

    if (show_help) {
        printUsage(argv[0]);
        return invalid_arg ? 1 : 0;
    }

    // If no mode specified, default to CLI
    if (!cli_mode && !gui_mode && !train_mode && !test_mode) {
        cli_mode = true;
    }

    try {
        // Initialize cloud storage
        auto cloud_manager = std::make_shared<CloudStorageManager>();
        setupCloudStorage(*cloud_manager);

        // Initialize memory compaction system
        auto memory_config = createDefaultMemoryConfig(cloud_manager, config_manager);
        auto memory_system = std::make_shared<MemoryCompactionSystem>(memory_config);

        // Set memory compaction system in HRM config
        hrm_config.memory_compaction_system = memory_system;

        std::shared_ptr<ResourceAwareHRM> hrm = nullptr;

        if (!test_mode) {
            // Initialize Vulkan resources
            VulkanResources vulkan;
            try {
                vulkan = initializeVulkan();
                logger.info("Vulkan initialized successfully");
            } catch (const std::exception& e) {
                logger.error("Failed to initialize Vulkan: " + std::string(e.what()));
                logger.warning("Falling back to CPU-only mode");
                // Continue without Vulkan - HRM will use CPU fallbacks
            }

            // Initialize HRM system only for CLI/GUI modes
            auto hrm_config = createDefaultHRMConfig(hw_caps);

            // Initialize HRM inner config - restore original layer counts
            hrm_config.base_config.base_config.hrm_config.inner_config.batch_size = 1;
            hrm_config.base_config.base_config.hrm_config.inner_config.seq_len = 512;
            hrm_config.base_config.base_config.hrm_config.inner_config.puzzle_emb_ndim = 128;
            hrm_config.base_config.base_config.hrm_config.inner_config.num_puzzle_identifiers = 10;
            hrm_config.base_config.base_config.hrm_config.inner_config.vocab_size = 100000;
            hrm_config.base_config.base_config.hrm_config.inner_config.H_cycles = 2;
            hrm_config.base_config.base_config.hrm_config.inner_config.L_cycles = 2;
            hrm_config.base_config.base_config.hrm_config.inner_config.H_layers = 4;
            hrm_config.base_config.base_config.hrm_config.inner_config.L_layers = 4;
            hrm_config.base_config.base_config.hrm_config.inner_config.hidden_size = 768;
            hrm_config.base_config.base_config.hrm_config.inner_config.expansion = 2.0f;
            hrm_config.base_config.base_config.hrm_config.inner_config.num_heads = 12;
            hrm_config.base_config.base_config.hrm_config.inner_config.pos_encodings = "learned";
            hrm_config.base_config.base_config.hrm_config.inner_config.rms_norm_eps = 1e-5f;
            hrm_config.base_config.base_config.hrm_config.inner_config.rope_theta = 10000.0f;
            hrm_config.base_config.base_config.hrm_config.inner_config.halt_max_steps = 8;  // Restore proper reasoning for research assistant
            hrm_config.base_config.base_config.hrm_config.inner_config.halt_exploration_prob = 0.1f;
            hrm_config.base_config.base_config.hrm_config.inner_config.forward_dtype = "float32";

    // Adapt configuration based on hardware capabilities
    adapt_config_to_hardware(hrm_config, hw_caps, vulkan);

    // Set Vulkan resources in config (only if Vulkan is available)
    if (vulkan.device != VK_NULL_HANDLE) {
        hrm_config.base_config.base_config.hrm_config.inner_config.physicalDevice = vulkan.physicalDevice;
        hrm_config.base_config.base_config.hrm_config.inner_config.device = vulkan.device;
        hrm_config.base_config.base_config.hrm_config.inner_config.computeQueue = vulkan.computeQueue;
        hrm_config.base_config.base_config.hrm_config.inner_config.computeQueueFamilyIndex = vulkan.computeQueueFamilyIndex;
        hrm_config.base_config.base_config.hrm_config.inner_config.commandPool = vulkan.commandPool;

        // Set Vulkan resources in UTF8 config
        hrm_config.base_config.base_config.utf8_config.physicalDevice = vulkan.physicalDevice;
        hrm_config.base_config.base_config.utf8_config.device = vulkan.device;
        hrm_config.base_config.base_config.utf8_config.computeQueue = vulkan.computeQueue;
        hrm_config.base_config.base_config.utf8_config.computeQueueFamilyIndex = vulkan.computeQueueFamilyIndex;
        hrm_config.base_config.base_config.utf8_config.commandPool = vulkan.commandPool;
    }

            hrm = std::make_shared<ResourceAwareHRM>(hrm_config);
            logger.info("HRM system initialized successfully");
        }

        if (test_mode) {
            logger.info("Running basic system tests");
            runBasicTests(hrm, memory_system, cloud_manager);
            logger.info("System tests completed");
            return 0;
        }

        if (train_mode) {
            logger.info("Starting HRM in Character Training mode");
            runCharacterTraining(hrm);
        } else if (cli_mode) {
            logger.info("Starting HRM in CLI mode");
            runCLI(hrm);
        } else if (gui_mode) {
            logger.info("Starting HRM in GUI mode");
            runGUI(hrm);
        }

    } catch (const std::exception& e) {
        logger.error("Error initializing HRM system: " + std::string(e.what()));
        return 1;
    }

    return 0;
}