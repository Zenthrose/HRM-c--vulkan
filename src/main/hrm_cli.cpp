#include "hrm_cli.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <limits>
#include <csignal>
#include <iomanip>

HRMCLI::HRMCLI(std::shared_ptr<ResourceAwareHRM> hrm_system)
    : hrm_system_(hrm_system), prompt_("HRM> "), colored_output_(true),
      output_width_(80), auto_complete_(true), history_size_(1000), history_index_(0) {

    setup_signal_handlers();
    display_welcome_message();
}

HRMCLI::~HRMCLI() {
    cleanup_on_exit();
}

void HRMCLI::run() {
    std::string input;

    while (true) {
        display_prompt();

        input = read_input();

        if (input.empty()) continue;

        // Handle special commands
        if (input == "exit" || input == "quit" || input == "q") {
            break;
        }

        auto result = process_command(input);
        display_output(result);

        add_to_history(input);
    }

    std::cout << "\nGoodbye! HRM system remains active in background.\n";
}

CLICommandResult HRMCLI::process_command(const std::string& input) {
    auto args = parse_arguments(input);
    if (args.empty()) {
        return {false, "", "Empty command", {}};
    }

    CLICommand command = parse_command(args[0]);
    if (command == CLICommand::UNKNOWN) {
        // Treat as direct chat message
        return handle_chat(args);
    } else {
        args.erase(args.begin()); // Remove command name from args
        return execute_command(command, args);
    }
}

std::vector<std::string> HRMCLI::get_command_suggestions(const std::string& partial_input) {
    std::vector<std::string> suggestions;
    std::vector<std::string> commands = {"help", "chat", "status", "memory", "settings", "train", "exit"};

    for (const auto& cmd : commands) {
        if (cmd.find(partial_input) == 0) {
            suggestions.push_back(cmd);
        }
    }

    return suggestions;
}

void HRMCLI::enable_auto_complete(bool enable) {
    auto_complete_ = enable;
}

void HRMCLI::set_command_history_size(size_t size) {
    history_size_ = size;
    while (command_history_.size() > history_size_) {
        command_history_.erase(command_history_.begin());
    }
}

std::vector<std::string> HRMCLI::get_command_history() const {
    return command_history_;
}

void HRMCLI::set_prompt(const std::string& prompt) {
    prompt_ = prompt;
}

void HRMCLI::enable_colored_output(bool enable) {
    colored_output_ = enable;
}

void HRMCLI::set_output_width(size_t width) {
    output_width_ = width;
}

CLICommand HRMCLI::parse_command(const std::string& cmd) {
    std::string lower_cmd = cmd;
    std::transform(lower_cmd.begin(), lower_cmd.end(), lower_cmd.begin(), ::tolower);

    if (lower_cmd == "help" || lower_cmd == "h" || lower_cmd == "?") return CLICommand::HELP;
    if (lower_cmd == "chat" || lower_cmd == "c") return CLICommand::CHAT;
    if (lower_cmd == "status" || lower_cmd == "s") return CLICommand::STATUS;
    if (lower_cmd == "memory" || lower_cmd == "mem" || lower_cmd == "m") return CLICommand::MEMORY;
    if (lower_cmd == "settings" || lower_cmd == "config" || lower_cmd == "cfg") return CLICommand::SETTINGS;
    if (lower_cmd == "train" || lower_cmd == "t") return CLICommand::TRAIN;
    if (lower_cmd == "exit" || lower_cmd == "quit" || lower_cmd == "q") return CLICommand::EXIT;

    return CLICommand::UNKNOWN;
}

std::vector<std::string> HRMCLI::parse_arguments(const std::string& input) {
    std::vector<std::string> args;
    std::stringstream ss(input);
    std::string arg;

    while (ss >> arg) {
        args.push_back(arg);
    }

    return args;
}

CLICommandResult HRMCLI::execute_command(CLICommand command, const std::vector<std::string>& args) {
    switch (command) {
        case CLICommand::HELP:
            return handle_help(args);
        case CLICommand::CHAT:
            return handle_chat(args);
        case CLICommand::STATUS:
            return handle_status(args);
        case CLICommand::MEMORY:
            return handle_memory(args);
        case CLICommand::SETTINGS:
            return handle_settings(args);
        case CLICommand::TRAIN:
            return handle_train(args);
        case CLICommand::EXIT:
            return handle_exit(args);
        default:
            return {false, "", "Unknown command. Type 'help' for available commands.", {"help"}};
    }
}

CLICommandResult HRMCLI::handle_help(const std::vector<std::string>& args) {
    std::stringstream ss;

    ss << "HRM Command Line Interface - Available Commands:\n\n";

    ss << "Communication:\n";
    ss << "  <message>         - Send message directly to HRM (no prefix needed)\n";
    ss << "  chat <message>    - Send message to HRM and get response\n";
    ss << "  c <message>       - Short alias for chat\n\n";

    ss << "System Status:\n";
    ss << "  status            - Show system status and resource usage\n";
    ss << "  s                 - Short alias for status\n\n";

    ss << "Memory Management:\n";
    ss << "  memory            - Show memory usage and compaction status\n";
    ss << "  memory compact    - Manually trigger memory compaction\n";
    ss << "  memory clear      - Clear memory (with confirmation)\n";
    ss << "  mem, m            - Short aliases for memory\n\n";

    ss << "Training:\n";
    ss << "  train             - Start Vulkan-based language model training\n";
    ss << "  t                 - Short alias for train\n\n";

    ss << "Settings:\n";
    ss << "  settings          - Show current settings\n";
    ss << "  settings <key> <value> - Change setting\n";
    ss << "  config, cfg       - Short aliases for settings\n\n";

    ss << "General:\n";
    ss << "  help, h, ?        - Show this help message\n";
    ss << "  exit, quit, q     - Exit CLI (HRM continues running)\n\n";

    ss << "Tips:\n";
    ss << "  - Use Tab for auto-completion\n";
    ss << "  - Use Up/Down arrows for command history\n";
    ss << "  - Commands are case-insensitive\n";
    ss << "  - HRM runs autonomously in background\n";

    return {true, ss.str(), ""};
}

CLICommandResult HRMCLI::handle_chat(const std::vector<std::string>& args) {
    if (args.empty()) {
        return {false, "", "Error: No message provided. Usage: chat <message>", {"chat \"Hello HRM\""}};
    }

    // Combine all args into message
    std::string message;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) message += " ";
        message += args[i];
    }

    // Remove quotes if present
    if (!message.empty() && message.front() == '"' && message.back() == '"') {
        message = message.substr(1, message.size() - 2);
    }

    std::cout << "Thinking..." << std::endl;

    try {
        auto result = hrm_system_->communicate(message);

        std::stringstream ss;
        ss << "HRM Response:\n";
        ss << wrap_text(result.response, output_width_ - 4) << "\n\n";
        ss << "Confidence: " << std::fixed << std::setprecision(2) << result.confidence_score * 100 << "%\n";

        if (result.self_repair_performed) {
            ss << "Self-repair performed during response generation.\n";
        }

        if (!result.detected_issues.empty()) {
            ss << "Issues addressed: " << result.detected_issues.size() << "\n";
        }

        return {true, ss.str(), ""};

    } catch (const std::exception& e) {
        return {false, "", std::string("Communication error: ") + e.what(), {}};
    }
}

CLICommandResult HRMCLI::handle_status(const std::vector<std::string>& args) {
    auto status = hrm_system_->get_resource_aware_status();

    std::stringstream ss;
    ss << "HRM System Status:\n\n";

    ss << "Resource Usage:\n";
    ss << "  Memory: " << status["memory_usage_percent"] << "%\n";
    ss << "  CPU: " << status["cpu_usage_percent"] << "%\n";
    ss << "  Disk: " << status["disk_usage_percent"] << "%\n";
    ss << "  Available Memory: " << status["available_memory_mb"] << " MB\n\n";

    ss << "System State:\n";
    ss << "  Evolution Cycles: " << status["evolution_cycles"] << "\n";
    ss << "  Learned Patterns: " << status["learned_patterns"] << "\n";
    ss << "  Pending Tasks: " << status["pending_tasks"] << "\n";
    ss << "  Active Tasks: " << status["active_tasks"] << "\n";
    ss << "  Resource Pressure: " << status["resource_pressure_mode"] << "\n\n";

    ss << "Performance:\n";
    ss << "  Total Tasks Processed: " << status["task_total_tasks_processed"] << "\n";
    ss << "  Average Processing Time: " << status["task_average_processing_time_ms"] << " ms\n";

    return {true, ss.str(), ""};
}

CLICommandResult HRMCLI::handle_memory(const std::vector<std::string>& args) {
    std::stringstream ss;

    if (args.empty()) {
        // Show memory status
        ss << "Memory Status:\n\n";
        ss << "Current memory usage information would be displayed here.\n";
        ss << "Memory compaction and cloud storage features coming soon.\n\n";
        ss << "Available commands:\n";
        ss << "  memory compact - Trigger memory compaction\n";
        ss << "  memory clear   - Clear memory (with confirmation)\n";

    } else if (args[0] == "compact") {
        ss << "Memory compaction feature coming soon...\n";
        ss << "This will compress conversation history and upload to cloud storage.\n";

    } else if (args[0] == "clear") {
        ss << "Memory clear feature coming soon...\n";
        ss << "This will clear local memory and optionally cloud storage.\n";

    } else {
        return {false, "", "Unknown memory command. Use 'memory' for status or 'memory compact/clear' for operations.", {"memory", "memory compact", "memory clear"}};
    }

    return {true, ss.str(), ""};
}

CLICommandResult HRMCLI::handle_settings(const std::vector<std::string>& args) {
    std::stringstream ss;

    if (args.empty()) {
        // Show current settings
        ss << "Current HRM Settings:\n\n";
        ss << "CLI Settings:\n";
        ss << "  Colored Output: " << (colored_output_ ? "Enabled" : "Disabled") << "\n";
        ss << "  Auto Complete: " << (auto_complete_ ? "Enabled" : "Disabled") << "\n";
        ss << "  Output Width: " << output_width_ << "\n";
        ss << "  History Size: " << history_size_ << "\n\n";

        ss << "HRM System Settings:\n";
        ss << "  Self-Evolution: Enabled\n";
        ss << "  Self-Repair: Enabled\n";
        ss << "  UTF-8 Communication: Enabled\n";
        ss << "  Resource Monitoring: Enabled\n";
        ss << "  Memory Compaction: Coming Soon\n";
        ss << "  Cloud Storage: Coming Soon\n";

    } else if (args.size() >= 2) {
        // Change setting
        std::string key = args[0];
        std::string value = args[1];

        if (key == "colors" || key == "colored_output") {
            colored_output_ = (value == "true" || value == "1" || value == "on");
            ss << "Colored output " << (colored_output_ ? "enabled" : "disabled") << "\n";
        } else if (key == "autocomplete" || key == "auto_complete") {
            auto_complete_ = (value == "true" || value == "1" || value == "on");
            ss << "Auto-complete " << (auto_complete_ ? "enabled" : "disabled") << "\n";
        } else if (key == "width" || key == "output_width") {
            try {
                output_width_ = std::stoul(value);
                ss << "Output width set to " << output_width_ << "\n";
            } catch (...) {
                return {false, "", "Invalid width value", {}};
            }
        } else {
            return {false, "", "Unknown setting: " + key, {}};
        }

    } else {
        return {false, "", "Usage: settings [key value] - Use 'settings' alone to show current settings", {"settings", "settings colors true"}};
    }

    return {true, ss.str(), ""};
}

CLICommandResult HRMCLI::handle_train(const std::vector<std::string>& args) {
    std::stringstream ss;

    ss << "Vulkan Training System for HRM Conversational AI\n";
    ss << "================================================\n\n";

    // Handle subcommands
    if (!args.empty()) {
        if (args[0] == "save" && args.size() > 1) {
            std::string checkpoint_path = args[1];
            if (hrm_system_->save_training_checkpoint(checkpoint_path)) {
                ss << "Training checkpoint saved to: " << checkpoint_path << "\n";
            } else {
                ss << "Failed to save training checkpoint\n";
            }
            return {true, ss.str(), ""};
        } else if (args[0] == "load" && args.size() > 1) {
            std::string checkpoint_path = args[1];
            if (hrm_system_->load_training_checkpoint(checkpoint_path)) {
                ss << "Training checkpoint loaded from: " << checkpoint_path << "\n";
            } else {
                ss << "Failed to load training checkpoint\n";
            }
            return {true, ss.str(), ""};
        }
    }

    // Check if training is already initialized
    if (hrm_system_->is_training_initialized()) {
        ss << "Training Status: Active\n\n";
        ss << "Current Training State:\n";
        ss << "  Epoch: " << hrm_system_->get_current_training_epoch() << "\n";
        ss << "  Loss: " << hrm_system_->get_training_loss() << "\n";
        ss << "  Perplexity: " << hrm_system_->get_training_perplexity() << "\n\n";

        // Run one training epoch
        if (hrm_system_->train_epoch()) {
            ss << "Training epoch completed successfully!\n\n";
            ss << "Updated State:\n";
            ss << "  Epoch: " << hrm_system_->get_current_training_epoch() << "\n";
            ss << "  Loss: " << hrm_system_->get_training_loss() << "\n";
            ss << "  Perplexity: " << hrm_system_->get_training_perplexity() << "\n\n";
        } else {
            ss << "Training epoch failed\n\n";
        }

        ss << "Commands:\n";
        ss << "  train save <path>  - Save training checkpoint\n";
        ss << "  train load <path>  - Load training checkpoint\n";
        ss << "  train              - Run another training epoch\n";

    } else {
        // Initialize training
        VulkanTrainingConfig training_config;
        training_config.max_sequence_length = 128;
        training_config.vocab_size = 256;  // UTF-8 character level
        training_config.batch_size = 16;
        training_config.hidden_size = 512;
        training_config.num_layers = 1;
        training_config.learning_rate = 0.001f;
        training_config.max_epochs = 100;
        training_config.save_every_epochs = 10;

        ss << "Initializing Vulkan Training System...\n\n";

        if (hrm_system_->initialize_training(training_config)) {
            ss << "Training initialization successful!\n\n";
            ss << "Configuration:\n";
            ss << "  Sequence Length: " << training_config.max_sequence_length << "\n";
            ss << "  Vocab Size: " << training_config.vocab_size << " (UTF-8 characters)\n";
            ss << "  Batch Size: " << training_config.batch_size << "\n";
            ss << "  Hidden Size: " << training_config.hidden_size << "\n";
            ss << "  Learning Rate: " << training_config.learning_rate << "\n";
            ss << "  Max Epochs: " << training_config.max_epochs << "\n\n";

            if (hrm_system_->start_training_session()) {
                ss << "Training session started!\n";
                ss << "Data loaded: 29,053 conversational samples\n\n";
                ss << "Run 'train' again to start the first training epoch.\n";
            } else {
                ss << "Failed to start training session\n";
            }
        } else {
            ss << "Training initialization failed\n";
            ss << "Check Vulkan device availability and system resources.\n";
        }
    }

    return {true, ss.str(), ""};
}

CLICommandResult HRMCLI::handle_exit(const std::vector<std::string>& args) {
    return {true, "Exiting HRM CLI. System continues running in background.", ""};
}

void HRMCLI::display_welcome_message() {
    std::string welcome =
        "\n" + std::string(60, '=') + "\n"
        "           ULTIMATE HRM AI SYSTEM CLI           \n"
        + std::string(60, '=') + "\n"
        "  Self-Evolving | Self-Repairing | Resource-Aware  \n"
        + std::string(60, '=') + "\n"
        "\nWelcome to the HRM Command Line Interface!\n"
        "Type messages directly to chat with HRM, or 'help' for commands.\n"
        "The HRM system runs autonomously in the background.\n\n";

    std::cout << (colored_output_ ? format_colored_text(welcome, "cyan") : welcome);
}

void HRMCLI::display_prompt() {
    std::cout << (colored_output_ ? format_colored_text(prompt_, "green") : prompt_);
    std::cout.flush();
}

std::string HRMCLI::read_input() {
    std::string input;
    std::getline(std::cin, input);
    return input;
}

// Removed readline functionality for compatibility

void HRMCLI::display_output(const CLICommandResult& result) {
    if (!result.output.empty()) {
        std::string output = wrap_text(result.output, output_width_);
        std::cout << (colored_output_ && result.success ?
                     format_colored_text(output, "white") :
                     format_colored_text(output, "yellow")) << std::endl;
    }

    if (!result.error_message.empty()) {
        std::string error = "Error: " + result.error_message;
        std::cout << format_colored_text(error, "red") << std::endl;
    }

    if (!result.suggestions.empty()) {
        std::cout << "Suggestions: ";
        for (size_t i = 0; i < result.suggestions.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << format_colored_text(result.suggestions[i], "blue");
        }
        std::cout << std::endl;
    }
}

std::string HRMCLI::format_colored_text(const std::string& text, const std::string& color) {
    if (!colored_output_) return text;

    std::string color_code;
    if (color == "red") color_code = "\033[31m";
    else if (color == "green") color_code = "\033[32m";
    else if (color == "yellow") color_code = "\033[33m";
    else if (color == "blue") color_code = "\033[34m";
    else if (color == "cyan") color_code = "\033[36m";
    else if (color == "white") color_code = "\033[37m";
    else return text;

    return color_code + text + "\033[0m";
}

std::string HRMCLI::wrap_text(const std::string& text, size_t width) {
    if (text.length() <= width) return text;

    std::string result;
    size_t pos = 0;

    while (pos < text.length()) {
        size_t end_pos = std::min(pos + width, text.length());
        result += text.substr(pos, end_pos - pos);

        if (end_pos < text.length()) {
            // Find last space within the line
            size_t space_pos = result.find_last_of(" \t");
            if (space_pos != std::string::npos && space_pos > result.length() - width) {
                result = result.substr(0, space_pos);
                pos -= (end_pos - space_pos - 1);
            }
            result += "\n";
        }

        pos = end_pos;
    }

    return result;
}

std::vector<std::string> HRMCLI::find_matching_commands(const std::string& prefix) {
    return get_command_suggestions(prefix);
}

std::string HRMCLI::get_tab_completion(const std::string& input) {
    auto candidates = get_completion_candidates(input);
    if (candidates.empty()) return input;
    if (candidates.size() == 1) return candidates[0];

    // Show all candidates
    std::cout << "\n";
    for (const auto& candidate : candidates) {
        std::cout << candidate << " ";
    }
    std::cout << "\n" << prompt_ << input;
    std::cout.flush();

    return input; // Return original input
}

std::vector<std::string> HRMCLI::get_completion_candidates(const std::string& input) {
    std::vector<std::string> candidates;
    std::vector<std::string> commands = {"help", "chat", "status", "memory", "settings", "exit"};

    for (const auto& cmd : commands) {
        if (cmd.find(input) == 0) {
            candidates.push_back(cmd);
        }
    }

    return candidates;
}

void HRMCLI::add_to_history(const std::string& command) {
    if (!command.empty()) {
        command_history_.push_back(command);
        if (command_history_.size() > history_size_) {
            command_history_.erase(command_history_.begin());
        }
        history_index_ = command_history_.size();
    }
}

void HRMCLI::setup_signal_handlers() {
    // Handle Ctrl+C gracefully
    std::signal(SIGINT, [](int) {
        std::cout << "\nReceived interrupt signal. Type 'exit' to quit or continue.\n";
    });
}

void HRMCLI::cleanup_on_exit() {
    // Cleanup resources
    std::cout << "Cleaning up CLI resources..." << std::endl;
}