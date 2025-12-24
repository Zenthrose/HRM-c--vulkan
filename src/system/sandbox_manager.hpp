#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <functional>
#include "../self_mod/code_analysis_system.hpp"

struct TestResult {
    bool success;
    double execution_time_seconds;
    double memory_usage_mb;
    double cpu_usage_percent;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    std::unordered_map<std::string, double> performance_metrics;
    std::string test_output;
};

struct ValidationResult {
    bool approved;
    double confidence_score; // 0.0 to 1.0
    double risk_assessment; // 0.0 (safe) to 1.0 (high risk)
    std::vector<std::string> concerns;
    std::vector<std::string> recommendations;
    std::string validation_report;
};

struct DeploymentDecision {
    bool deploy;
    std::string reasoning;
    std::vector<std::string> conditions;
    std::chrono::seconds monitoring_period;
};

class SandboxManager {
public:
    SandboxManager();
    ~SandboxManager();

    // Main testing interface
    TestResult test_modification(const CodeModification& modification);
    ValidationResult validate_modification(const TestResult& test_result);
    DeploymentDecision make_deployment_decision(const ValidationResult& validation);

    // Sandbox management
    bool initialize_sandbox();
    bool cleanup_sandbox();
    bool is_sandbox_ready() const;

    // Configuration
    void set_resource_limits(double max_cpu_percent, double max_memory_mb, std::chrono::seconds max_time);
    void set_test_scenarios(const std::vector<std::string>& scenarios);
    void set_validation_criteria(double min_confidence, double max_risk);

private:
    // Sandbox components
    class ProcessSandbox;
    class MemorySandbox;
    class FileSystemSandbox;

    std::unique_ptr<ProcessSandbox> process_sandbox_;
    std::unique_ptr<MemorySandbox> memory_sandbox_;
    std::unique_ptr<FileSystemSandbox> fs_sandbox_;

    // Configuration
    double max_cpu_percent_;
    double max_memory_mb_;
    std::chrono::seconds max_test_time_;
    std::vector<std::string> test_scenarios_;
    double min_confidence_threshold_;
    double max_risk_threshold_;

    // Test execution
    TestResult run_performance_tests(const CodeModification& mod);
    TestResult run_functionality_tests(const CodeModification& mod);
    TestResult run_safety_tests(const CodeModification& mod);
    TestResult run_regression_tests(const CodeModification& mod);

    // Validation helpers
    double calculate_confidence_score(const TestResult& result);
    double assess_risk_level(const TestResult& result);
    std::vector<std::string> identify_concerns(const TestResult& result);
    std::vector<std::string> generate_recommendations(const ValidationResult& validation);

    // Utility functions
    std::string generate_test_executable(const CodeModification& mod);
    bool compile_test_code(const std::string& source_path, const std::string& output_path);
    TestResult execute_test(const std::string& test_executable);
    void log_test_activity(const std::string& activity, const std::string& details);
};