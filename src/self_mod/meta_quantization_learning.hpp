#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include <functional>
#include <chrono>

// Include quantization type definitions
#include "../vulkan/quantization_types.hpp"

#include <random>
namespace Nyx {

// Forward declarations used below
struct EvolutionResults;
struct QuantizationImprovement;
struct PerformanceGains;
struct QuantizationStrategy;
struct QuantizationRule;
struct CodeModification;
struct SystemPerformanceMetrics;

// Quantization experiment forward declarations
class QuantizationExperiment;
class AdaptiveQuantizationManager;

// Meta-learning for quantization strategies
class MetaQuantizationLearner {
public:
    MetaQuantizationLearner();

    // Learn from quantization experiments
    void learn_from_experiment(const QuantizationExperiment& experiment,
                              const QuantizationPerformance& performance);

    // Predict optimal quantization for new scenarios
    QuantizationConfig predict_optimal_config(const HardwareCapabilities& hw,
                                            const TaskRequirements& task);

    // Update meta-knowledge with new data
    void update_meta_knowledge(const std::vector<QuantizationExperiment>& experiments);

    // Get confidence in predictions
    float get_prediction_confidence(const QuantizationConfig& config) const;

private:
    // Meta-learning model state
    std::unordered_map<std::string, QuantizationPerformance> performance_database_;
    std::unordered_map<std::string, QuantizationConfig> optimal_configs_;
    std::vector<std::pair<HardwareCapabilities, QuantizationConfig>> meta_examples_;

    // RNG used by the learner
    std::mt19937 rng_;

    // Counters and stats used by implementation
    int low_memory_int4_success_ = 0;
    int tensor_core_fp16_success_ = 0;
    int successful_experiments_ = 0;

    // Private helper methods implemented in .cpp
    QuantizationConfig get_conservative_defaults(const HardwareCapabilities& hw);
    float calculate_hardware_similarity(const HardwareCapabilities& hw1, const HardwareCapabilities& hw2) const;
    std::string generate_experiment_key(const QuantizationExperiment& exp) const;
    std::string generate_hardware_key(const HardwareCapabilities& hw) const;
    QuantizationPerformance get_performance_from_experiment(const QuantizationExperiment& exp) const;

    // Learning algorithms
    void train_meta_model();
    HardwareCapabilities find_similar_hardware(const HardwareCapabilities& hw) const;
    void update_similarity_metrics(const QuantizationExperiment& exp);
};

// Quantization experiment data
struct QuantizationExperiment {
    std::string experiment_id;
    HardwareCapabilities hardware;
    TaskRequirements task;
    QuantizationConfig config_used;
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;

    // Experiment metadata
    std::string model_type;
    size_t original_model_size;
    std::unordered_map<std::string, float> experiment_metadata;
};

// Evolutionary quantization optimizer
class EvolutionaryQuantizationOptimizer {
public:
    EvolutionaryQuantizationOptimizer(std::shared_ptr<MetaQuantizationLearner> meta_learner);

    // Evolutionary search for optimal quantization
    std::vector<QuantizationConfig> evolve_quantization_configs(
        const HardwareCapabilities& hw,
        const TaskRequirements& task,
        size_t population_size = 20,
        size_t generations = 10);

    // Fitness evaluation
    float evaluate_config_fitness(const QuantizationConfig& config,
                                const HardwareCapabilities& hw,
                                const QuantizationPerformance& performance);

    // Mutation and crossover operators
    QuantizationConfig mutate_config(const QuantizationConfig& config);
    std::pair<QuantizationConfig, QuantizationConfig> crossover_configs(
        const QuantizationConfig& parent1, const QuantizationConfig& parent2);

private:
    std::shared_ptr<MetaQuantizationLearner> meta_learner_;
    std::mt19937 rng_;

    // Evolution parameters
    float mutation_rate_ = 0.1f;
    float crossover_rate_ = 0.8f;

    // Population management
    std::vector<QuantizationConfig> initialize_population(size_t size,
                                                        const HardwareCapabilities& hw);
    std::vector<QuantizationConfig> select_best(const std::vector<QuantizationConfig>& population,
                                              const std::vector<float>& fitness_scores,
                                              size_t keep_count);
};

// System performance structure used by self-evolving component
struct SystemPerformanceMetrics {
    float accuracy;
    float memory_efficiency;
    float latency_ms;
    float energy_consumption;
};

// Self-evolving quantization system
class SelfEvolvingQuantization {
public:
    SelfEvolvingQuantization(std::shared_ptr<AdaptiveQuantizationManager> quant_manager,
                           std::shared_ptr<MetaQuantizationLearner> meta_learner);

    // Continuous self-improvement
    void evolve_quantization_system();

    // Learn from system performance
    void learn_from_system_performance(const SystemPerformanceMetrics& metrics);

    // Self-modify quantization strategies
    void apply_self_modifications(const EvolutionResults& results);

    // Meta-evolution of the quantization system itself
    void meta_evolve_quantization_system();

private:
    std::shared_ptr<AdaptiveQuantizationManager> quantization_manager_;
    std::shared_ptr<MetaQuantizationLearner> meta_learner_;

    // Self-evolution state
    std::vector<EvolutionResults> evolution_history_;
    // Map improvement type -> canonical QuantizationConfig learned
    std::unordered_map<std::string, QuantizationConfig> learned_strategies_;
    
    // Counters and helper state used by implementation
    int successful_evolution_cycles_ = 0;
    float evolution_aggressiveness_ = 1.0f;
    float evolution_frequency_ = 1.0f;

    // Functions implemented in .cpp
    std::vector<std::string> detect_evolution_patterns();
    std::vector<std::string> generate_meta_improvements(const std::vector<std::string>& patterns);
    void apply_meta_evolution(const std::string& improvement);
    void apply_meta_evolution();
    void adapt_evolution_parameters();

    // Evolution methods
    EvolutionResults analyze_current_performance();
    std::vector<CodeModification> generate_quantization_improvements();
    void validate_and_apply_evolution(const EvolutionResults& results);
};


// Performance gains summary for an evolution attempt
struct PerformanceGains {
    float accuracy_improvement = 0.0f;
    float speed_improvement = 0.0f;
    size_t memory_reduction = 0;
    float energy_efficiency_gain = 0.0f;
};

// Evolution results and tracking
struct EvolutionResults {
    std::string evolution_id;
    std::chrono::system_clock::time_point timestamp;
    std::vector<QuantizationImprovement> improvements;
    PerformanceGains gains;
    std::vector<std::string> applied_modifications;
};

struct QuantizationImprovement {
    std::string improvement_type;
    std::string description;
    float performance_gain;
    QuantizationConfig affected_config;
};

// Simple code modification descriptor used by the meta-evolver
struct CodeModification {
    std::string file_path;
    int start_line = 0;
    int end_line = 0;
    std::string original_code;
    std::string new_code;
    std::string reason;
    std::string description;
};

// Quantization strategy representation
struct QuantizationStrategy {
    std::string strategy_id;
    std::string strategy_name;
    std::vector<QuantizationRule> rules;
    std::unordered_map<std::string, float> strategy_metadata;

    // Strategy application
    QuantizationConfig apply_to_hardware(const HardwareCapabilities& hw,
                                       const TaskRequirements& task) const;
};

struct QuantizationRule {
    std::string condition_type;  // "hardware", "task", "performance"
    std::string condition_value;
    QuantizationConfig resulting_config;
    float confidence_score;
};

} // namespace Nyx