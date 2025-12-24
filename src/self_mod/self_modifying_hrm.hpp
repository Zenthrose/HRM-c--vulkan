#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "self_evolving_hrm.hpp"
#include "code_analysis_system.hpp"
#include "runtime_compilation_system.hpp"
#include "../system/sandbox_manager.hpp"
#include <iostream>

struct SelfModifyingHRMConfig {
    SelfEvolvingHRMConfig base_config;
    std::string project_root;
    std::string temp_compilation_dir;
    bool enable_self_modification;
    bool enable_runtime_recompilation;
    float self_analysis_frequency; // How often to analyze own code (in interactions)
    float modification_confidence_threshold; // Minimum confidence to apply modifications
    bool create_backups_before_modification;
    std::vector<std::string> protected_files; // Files that should not be modified
};

struct CodeChange {
    std::string file_path;
    std::string old_code;
    std::string new_code;
};

struct SelfModificationResult {
    bool modification_applied;
    std::string modified_file;
    std::string modification_description;
    float confidence_score;
    bool compilation_successful;
    bool system_restart_required;
    std::vector<std::string> potential_risks;
    std::string rollback_instructions;
    std::vector<CodeChange> code_changes;
};

class SelfModifyingHRM : public SelfEvolvingHRM {
public:
    SelfModifyingHRM(const SelfModifyingHRMConfig& config);
    ~SelfModifyingHRM();

    // Enhanced communication with self-analysis
    CommunicationResult communicate(const std::string& input_message);

    // Self-modification methods
    SelfModificationResult analyze_and_modify_self();
    bool apply_self_modification(const SelfModificationResult& modification);
    bool rollback_self_modification(const std::string& backup_id);

    // System introspection
    std::unordered_map<std::string, std::string> get_self_analysis_report();
    std::vector<std::string> detect_self_limitations();
    std::vector<std::string> propose_self_improvements();

    // Safety and monitoring
    bool validate_self_modification_safety(const SelfModificationResult& modification);
    void log_self_modification_activity(const SelfModificationResult& modification);
    std::vector<std::string> get_self_modification_history();

    // Additional methods for service integration
    void add_boot_task(const std::function<void()>& task);
    void add_idle_task(const std::function<void()>& task);
    bool should_analyze_self() const;

    // Enhanced safety for hot-swaps
    bool validate_hot_swap_safety(const std::string& file_path, const std::string& new_code);
    bool validate_code_syntax(const std::string& code);
    bool create_safety_checkpoint(const std::string& description);
    bool restore_from_safety_checkpoint(const std::string& checkpoint_id);
    std::vector<std::string> scan_code_for_risks(const std::string& code_content);
    bool validate_code_integrity(const std::string& file_path, const std::string& expected_hash);
    bool perform_dynamic_validation(const SelfModificationResult& modification);
    bool validate_system_stability();

private:
    SelfModifyingHRMConfig config_;
    std::unique_ptr<CodeAnalysisSystem> code_analyzer_;
    std::unique_ptr<RuntimeCompilationSystem> runtime_compiler_;
    std::unique_ptr<SandboxManager> sandbox_manager_;

    // Self-modification state
    int interactions_since_last_analysis_;
    std::vector<SelfModificationResult> modification_history_;
    std::unordered_map<std::string, std::string> active_backups_;
    std::vector<std::function<void()>> boot_tasks_;
    std::vector<std::function<void()>> idle_tasks_;
    std::unordered_map<std::string, std::string> safety_checkpoints_;
    std::vector<std::string> protected_system_files_;

    // Hot-swappable function for demonstration
    typedef void (*ModifiableFunction)(const std::string&);
    ModifiableFunction current_modifiable_function;
    void* loaded_module_handle;

    static void default_modifiable_function(const std::string& message) {
        std::cout << "[DEFAULT] " << message << std::endl;
    }

    // Core self-modification logic
    SelfModificationResult perform_self_analysis();
    std::vector<CodeModification> generate_self_fixes(const CodeAnalysisResult& analysis);
    SelfModificationResult evaluate_modification_impact(const std::vector<CodeModification>& modifications);

    // Safety checks
    bool is_file_modification_safe(const std::string& file_path);
    bool validate_modification_semantics(const CodeModification& modification);
    float assess_modification_risk(const CodeModification& modification);

    // Compilation and deployment
    bool compile_modified_system(const std::vector<CodeModification>& modifications);
    bool hot_swap_modified_components();
    void update_system_configuration();

    // Fallback and recovery
    bool create_system_backup();
    bool restore_system_backup(const std::string& backup_id);
    void enter_safe_mode();

    // Meta-analysis
    std::vector<std::string> analyze_self_improvement_opportunities();
    std::vector<std::string> detect_self_degradation_patterns();
    void adapt_self_analysis_parameters();
};