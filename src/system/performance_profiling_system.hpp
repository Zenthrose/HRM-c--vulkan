#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <chrono>
#include <memory>
#include <functional>

namespace Nyx {

// Performance metrics for quantization operations
struct QuantizationPerformanceMetrics {
    std::string operation_name;
    PrecisionLevel precision_level;
    double execution_time_ms;
    size_t memory_usage_mb;
    double accuracy_degradation;  // percentage vs FP32
    size_t model_size_mb;
    double throughput_items_per_sec;
    double power_consumption_watts;
    std::chrono::system_clock::time_point timestamp;
    std::unordered_map<std::string, double> custom_metrics;
};

// Comprehensive performance profiler
class QuantizationPerformanceProfiler {
public:
    QuantizationPerformanceProfiler();

    // Profile quantization operations
    void start_profiling(const std::string& operation_name, PrecisionLevel precision);
    void end_profiling(const std::string& operation_name);

    // Accuracy profiling
    void measure_accuracy_degradation(const Model& fp32_model,
                                    const QuantizedModel& quantized_model,
                                    const std::vector<Tensor>& test_data);

    // Memory profiling
    void profile_memory_usage(const QuantizedModel& model,
                            const HardwareCapabilities& hw);

    // Performance analysis
    PerformanceAnalysis analyze_performance() const;
    std::vector<OptimizationRecommendation> generate_recommendations() const;

    // Benchmarking utilities
    BenchmarkResult benchmark_precision(PrecisionLevel precision,
                                      const Model& model,
                                      const HardwareCapabilities& hw);

    // Export profiling data
    void export_profiling_data(const std::string& filename) const;

private:
    std::unordered_map<std::string, QuantizationPerformanceMetrics> metrics_;
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> active_profilers_;

    // Internal profiling methods
    void collect_hardware_metrics(QuantizationPerformanceMetrics& metrics);
    void calculate_derived_metrics(QuantizationPerformanceMetrics& metrics);
    double measure_power_consumption();
    size_t measure_memory_usage();
};

// Performance analysis and recommendations
struct PerformanceAnalysis {
    double average_accuracy_degradation;
    double average_speedup;
    double average_memory_reduction;
    std::vector<PrecisionLevel> optimal_precisions;
    std::unordered_map<PrecisionLevel, double> precision_scores;
    std::vector<std::string> performance_bottlenecks;
};

struct OptimizationRecommendation {
    std::string recommendation_type;  // "precision", "hardware", "algorithm"
    std::string description;
    double expected_improvement;
    PriorityLevel priority;
    std::vector<std::string> implementation_steps;
};

enum class PriorityLevel { LOW, MEDIUM, HIGH, CRITICAL };

// Benchmarking results
struct BenchmarkResult {
    PrecisionLevel precision;
    double average_latency_ms;
    double throughput_items_per_sec;
    double memory_usage_mb;
    double accuracy_score;
    double power_efficiency;  // items_per_watt
    std::vector<double> latency_distribution;  // percentiles
    std::chrono::system_clock::time_point benchmark_time;
};

// Real-time monitoring system
class QuantizationMonitoringSystem {
public:
    QuantizationMonitoringSystem();

    // Real-time monitoring
    void start_monitoring();
    void stop_monitoring();
    MonitoringSnapshot get_current_snapshot() const;

    // Alert system
    void set_performance_thresholds(const PerformanceThresholds& thresholds);
    std::vector<PerformanceAlert> check_for_alerts() const;

    // Historical analysis
    PerformanceTrends analyze_trends(std::chrono::hours lookback_period) const;
    std::vector<PerformanceAnomaly> detect_anomalies() const;

private:
    bool monitoring_active_;
    std::vector<MonitoringSnapshot> snapshots_;
    PerformanceThresholds thresholds_;

    // Background monitoring thread
    void monitoring_thread_function();
    void collect_system_metrics(MonitoringSnapshot& snapshot);
};

// Monitoring data structures
struct MonitoringSnapshot {
    std::chrono::system_clock::time_point timestamp;
    double cpu_usage_percent;
    double gpu_usage_percent;
    double memory_usage_percent;
    double gpu_memory_usage_mb;
    double power_consumption_watts;
    std::unordered_map<PrecisionLevel, double> precision_performance;
    std::vector<ActiveQuantizationOperation> active_operations;
};

struct ActiveQuantizationOperation {
    std::string operation_id;
    PrecisionLevel precision;
    double progress_percent;
    double estimated_completion_time;
    size_t memory_usage_mb;
};

struct PerformanceThresholds {
    double max_latency_ms;
    double max_memory_usage_percent;
    double max_power_consumption_watts;
    double min_accuracy_score;
    double max_cpu_usage_percent;
    double max_gpu_usage_percent;
};

struct PerformanceAlert {
    AlertLevel level;
    std::string message;
    std::string recommendation;
    std::chrono::system_clock::time_point timestamp;
    std::unordered_map<std::string, double> metrics;
};

enum class AlertLevel { INFO, WARNING, ERROR, CRITICAL };

struct PerformanceTrends {
    std::vector<double> latency_trend;
    std::vector<double> memory_trend;
    std::vector<double> accuracy_trend;
    double latency_trend_slope;
    double memory_trend_slope;
    double accuracy_trend_slope;
    std::string overall_trend;  // "improving", "degrading", "stable"
};

struct PerformanceAnomaly {
    std::string anomaly_type;
    std::string description;
    double severity_score;  // 0.0 to 1.0
    std::chrono::system_clock::time_point detected_at;
    std::vector<std::string> potential_causes;
    std::vector<std::string> recommended_actions;
};

// Automated testing framework
class QuantizationTestingFramework {
public:
    QuantizationTestingFramework(std::shared_ptr<QuantizationPerformanceProfiler> profiler);

    // Comprehensive testing
    TestSuiteResult run_full_test_suite(const Model& model,
                                      const HardwareCapabilities& hw);

    // Individual tests
    AccuracyTestResult test_accuracy_preservation(const Model& fp32_model,
                                                const QuantizedModel& quantized_model);

    PerformanceTestResult test_performance_regression(const Model& model,
                                                    PrecisionLevel precision);

    StabilityTestResult test_numerical_stability(const Model& model,
                                               PrecisionLevel precision);

    // Automated regression testing
    void setup_regression_tests(const std::vector<Model>& test_models);
    RegressionReport run_regression_tests();

private:
    std::shared_ptr<QuantizationPerformanceProfiler> profiler_;
    std::vector<Model> regression_test_models_;
};

// Test result structures
struct TestSuiteResult {
    bool overall_pass;
    std::vector<AccuracyTestResult> accuracy_tests;
    std::vector<PerformanceTestResult> performance_tests;
    std::vector<StabilityTestResult> stability_tests;
    std::string summary_report;
    std::vector<std::string> recommendations;
};

struct AccuracyTestResult {
    PrecisionLevel precision;
    double accuracy_score;
    double degradation_from_fp32;
    bool passes_threshold;
    std::vector<double> per_task_accuracy;
};

struct PerformanceTestResult {
    PrecisionLevel precision;
    double average_latency;
    double latency_regression;  // vs FP32
    double memory_usage;
    double throughput;
    bool passes_performance_targets;
};

struct StabilityTestResult {
    PrecisionLevel precision;
    bool numerically_stable;
    std::vector<std::string> stability_issues;
    double gradient_norm_stability;
    bool passes_stability_checks;
};

struct RegressionReport {
    int tests_run;
    int tests_passed;
    int tests_failed;
    std::vector<RegressionFailure> failures;
    std::string change_summary;
};

struct RegressionFailure {
    std::string test_name;
    std::string failure_reason;
    double previous_value;
    double current_value;
    double change_percentage;
};

// Forward declarations for missing types
struct Model;
struct QuantizedModel;
struct Tensor;
enum class PrecisionLevel;
struct HardwareCapabilities;

// Main monitoring and profiling system
class ComprehensiveQuantizationMonitor {
public:
    ComprehensiveQuantizationMonitor();

    // Initialize monitoring systems
    void initialize_monitoring(const HardwareCapabilities& hw);

    // Run comprehensive monitoring session
    MonitoringReport run_monitoring_session(const Model& model,
                                          const std::vector<QuantizationConfig>& configs,
                                          std::chrono::minutes duration);

    // Generate comprehensive reports
    QuantizationAnalysisReport generate_analysis_report() const;

    // Export data for further analysis
    void export_monitoring_data(const std::string& filename) const;

private:
    std::shared_ptr<QuantizationPerformanceProfiler> profiler_;
    std::shared_ptr<QuantizationMonitoringSystem> monitoring_system_;
    std::shared_ptr<QuantizationTestingFramework> testing_framework_;

    // Integrated analysis
    void correlate_performance_and_accuracy();
    void identify_system_bottlenecks();
    void generate_optimization_recommendations();
};

struct MonitoringReport {
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
    std::vector<MonitoringSnapshot> snapshots;
    std::vector<PerformanceAlert> alerts;
    PerformanceTrends trends;
    std::vector<PerformanceAnomaly> anomalies;
};

struct QuantizationAnalysisReport {
    PerformanceAnalysis overall_analysis;
    std::vector<OptimizationRecommendation> recommendations;
    std::vector<BenchmarkResult> benchmark_results;
    TestSuiteResult test_results;
    std::string executive_summary;
    std::vector<std::string> next_steps;
};

} // namespace Nyx