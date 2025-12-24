#pragma once

#include <string>
#include <vector>
#include <functional>
#include "../hrm/resource_aware_hrm.hpp"
#include "../physics/mcmc_simulator.hpp"
#include "../cad/freecad_interface.hpp"
#include "../self_mod/self_modifying_hrm.hpp"
#include "../vulkan/vulkan_trainer.hpp"

enum class CLICommand {
    HELP,
    CHAT,
    STATUS,
    MEMORY,
    SETTINGS,
    TRAIN,
    SEARCH,
    EDIT,
    BUILD,
    RUN,
    MCMC,
    CAD,
    OPTIMIZE,
    IDLE_LEARNING,
    EXIT,
    UNKNOWN
};

struct CLICommandResult {
    bool success;
    std::string output;
    std::string error_message;
    std::vector<std::string> suggestions;
};

class NyxCLI {
public:
    NyxCLI(std::shared_ptr<ResourceAwareHRM> hrm_system);
    ~NyxCLI();

    // Main CLI loop
    void run();

    // Command processing
    CLICommandResult process_command(const std::string& input);
    std::vector<std::string> get_command_suggestions(const std::string& partial_input);

    // Interactive features
    void enable_auto_complete(bool enable);
    void set_command_history_size(size_t size);
    std::vector<std::string> get_command_history() const;

    // UI customization
    void set_prompt(const std::string& prompt);
    void enable_colored_output(bool enable);
    void set_output_width(size_t width);

private:
    std::shared_ptr<ResourceAwareHRM> hrm_system_;
    std::string prompt_;
    bool colored_output_;
    size_t output_width_;
    bool auto_complete_;
    size_t history_size_;
    std::vector<std::string> command_history_;
    size_t history_index_;

    // Command parsing and execution
    CLICommand parse_command(const std::string& input);
    std::vector<std::string> parse_arguments(const std::string& input);
    CLICommandResult execute_command(CLICommand command, const std::vector<std::string>& args);

    // Individual command handlers
    CLICommandResult handle_help(const std::vector<std::string>& args);
    CLICommandResult handle_chat(const std::vector<std::string>& args);
    CLICommandResult handle_status(const std::vector<std::string>& args);
    CLICommandResult handle_memory(const std::vector<std::string>& args);
    CLICommandResult handle_settings(const std::vector<std::string>& args);
    CLICommandResult handle_train(const std::vector<std::string>& args);
    CLICommandResult execute_auto_training(const VulkanTrainingConfig& config);
    CLICommandResult handle_search(const std::vector<std::string>& args);
    CLICommandResult handle_edit(const std::vector<std::string>& args);
    CLICommandResult handle_build(const std::vector<std::string>& args);
    CLICommandResult handle_run(const std::vector<std::string>& args);
    CLICommandResult handle_mcmc(const std::vector<std::string>& args);
    CLICommandResult handle_cad(const std::vector<std::string>& args);
    CLICommandResult handle_optimize(const std::vector<std::string>& args);
    CLICommandResult handle_idle_learning(const std::vector<std::string>& args);
    CLICommandResult handle_exit(const std::vector<std::string>& args);

    // Helper functions
    void display_welcome_message();
    void display_prompt();
    std::string read_input();
    void display_output(const CLICommandResult& result);
    std::string format_colored_text(const std::string& text, const std::string& color);
    std::string wrap_text(const std::string& text, size_t width);
    std::vector<std::string> find_matching_commands(const std::string& prefix);

    // Tab completion
    std::string get_tab_completion(const std::string& input);
    std::vector<std::string> get_completion_candidates(const std::string& input);

    // History management
    void add_to_history(const std::string& command);
    std::string get_previous_command();
    std::string get_next_command();

    // System integration
    void setup_signal_handlers();
    void cleanup_on_exit();
};