#include "meta_quantization_learning.hpp"
#include "../vulkan/adaptive_quantization.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>

// QUANTIZATION_COMPONENT: Meta Quantization Learning - Nyx can modify or remove
// Enables Nyx to learn optimal quantization strategies while quantized
// Fallback: Use static quantization rules if meta-learning fails

namespace Nyx {

MetaQuantizationLearner::MetaQuantizationLearner()
    : rng_(std::random_device{}()) {
    std::cout << "Meta Quantization Learner initialized" << std::endl;
}

void MetaQuantizationLearner::learn_from_experiment(const QuantizationExperiment& experiment,
                                                  const QuantizationPerformance& performance) {
    // Learn from quantization experiment results
    std::string key = generate_experiment_key(experiment);

    // Store performance data
    performance_database_[key] = performance;

    // Update optimal configurations
    if (performance.accuracy_score >= 0.85f) { // Meets accuracy threshold
        optimal_configs_[key] = experiment.config_used;
    }

    // Add to meta-examples for learning
    HardwareCapabilities hw = experiment.hardware;
    TaskRequirements task = experiment.task;
    QuantizationConfig config = experiment.config_used;

    meta_examples_.push_back({hw, config});

    // Retrain meta-model if we have enough examples
    if (meta_examples_.size() % 10 == 0) { // Retrain every 10 examples
        train_meta_model();
    }

    std::cout << "Learned from quantization experiment: " << key << std::endl;
}

QuantizationConfig MetaQuantizationLearner::predict_optimal_config(const HardwareCapabilities& hw,
                                                                 const TaskRequirements& task) {
    // Predict optimal quantization config for new scenario

    if (meta_examples_.empty()) {
        // No learning data - return conservative defaults
        return get_conservative_defaults(hw);
    }

    // Find similar hardware configuration and lookup learned optimal config
    auto similar_hw = find_similar_hardware(hw);
    std::string key = generate_hardware_key(similar_hw);
    auto it = optimal_configs_.find(key);
    if (it != optimal_configs_.end()) {
        return it->second;
    }

    // Fallback to conservative defaults
    return get_conservative_defaults(hw);
}

void MetaQuantizationLearner::update_meta_knowledge(const std::vector<QuantizationExperiment>& experiments) {
    // Batch update meta-knowledge
    for (const auto& exp : experiments) {
        learn_from_experiment(exp, get_performance_from_experiment(exp));
    }

    // Update similarity metrics
    update_similarity_metrics(experiments.back());
}

float MetaQuantizationLearner::get_prediction_confidence(const QuantizationConfig& config) const {
    // Calculate confidence in prediction
    if (meta_examples_.empty()) return 0.0f;

    // Simple confidence based on number of similar examples
    int similar_count = 0;
    for (const auto& example : meta_examples_) {
        if (example.second.precision_level == config.precision_level) {
            similar_count++;
        }
    }

    return static_cast<float>(similar_count) / meta_examples_.size();
}

// Private methods
void MetaQuantizationLearner::train_meta_model() {
    // Train meta-model on collected examples
    // This is a simplified implementation - real version would use ML algorithms

    std::cout << "Training meta-model with " << meta_examples_.size() << " examples" << std::endl;

    // Simple rule-based learning for demonstration
    // In practice, this would use gradient descent or other ML algorithms
    for (const auto& example : meta_examples_) {
        const HardwareCapabilities& hw = example.first;
        const QuantizationConfig& config = example.second;

        // Learn associations between hardware and optimal precision
        if (hw.gpu_memory_mb < 4096 && config.precision_level == PrecisionLevel::INT4) {
            // Low memory GPUs work well with INT4
            low_memory_int4_success_++;
        }
        if (hw.has_tensor_cores && config.precision_level == PrecisionLevel::FP16) {
            // Tensor cores benefit from FP16
            tensor_core_fp16_success_++;
        }
    }

    std::cout << "Meta-model training complete" << std::endl;
}

HardwareCapabilities MetaQuantizationLearner::find_similar_hardware(const HardwareCapabilities& hw) const {
    // Find most similar hardware configuration from learned examples

    if (meta_examples_.empty()) return HardwareCapabilities{};

    HardwareCapabilities best_match = meta_examples_[0].first;
    float best_similarity = 0.0f;

    for (const auto& example : meta_examples_) {
        float similarity = calculate_hardware_similarity(hw, example.first);
        if (similarity > best_similarity) {
            best_similarity = similarity;
            best_match = example.first;
        }
    }

    return best_match;
}

void MetaQuantizationLearner::update_similarity_metrics(const QuantizationExperiment& exp) {
    // Update similarity calculation weights based on experiment results
    // This helps improve future hardware matching

    const QuantizationPerformance& perf = get_performance_from_experiment(exp);

    if (perf.accuracy_score >= 0.85f) {
        // Successful experiment - increase weight for similar hardware matching
        successful_experiments_++;
    }
}

QuantizationConfig MetaQuantizationLearner::get_conservative_defaults(const HardwareCapabilities& hw) {
    // Return conservative quantization defaults based on hardware

    QuantizationConfig config;
    config.precision_level = PrecisionLevel::FP32; // Start conservative
    config.per_channel_quantization = true;

    // Adjust based on hardware capabilities
    if (hw.supports_int8 && hw.gpu_memory_mb >= 4096) {
        config.precision_level = PrecisionLevel::INT8;
    } else if (hw.supports_fp16) {
        config.precision_level = PrecisionLevel::FP16;
    }

    return config;
}

float MetaQuantizationLearner::calculate_hardware_similarity(const HardwareCapabilities& hw1,
                                                          const HardwareCapabilities& hw2) const {
    // Calculate similarity score between two hardware configurations

    float similarity = 0.0f;
    float total_weight = 0.0f;

    // Memory similarity (weighted heavily)
    float mem_diff = std::abs(static_cast<int>(hw1.gpu_memory_mb) - static_cast<int>(hw2.gpu_memory_mb));
    float mem_similarity = 1.0f / (1.0f + mem_diff / 1024.0f); // Normalize by 1GB
    similarity += mem_similarity * 0.4f;
    total_weight += 0.4f;

    // Precision support similarity
    float precision_score = 0.0f;
    if (hw1.supports_fp16 == hw2.supports_fp16) precision_score += 0.25f;
    if (hw1.supports_int8 == hw2.supports_int8) precision_score += 0.25f;
    if (hw1.supports_int4 == hw2.supports_int4) precision_score += 0.25f;
    if (hw1.has_tensor_cores == hw2.has_tensor_cores) precision_score += 0.25f;
    similarity += precision_score * 0.6f;
    total_weight += 0.6f;

    return similarity / total_weight;
}

std::string MetaQuantizationLearner::generate_experiment_key(const QuantizationExperiment& exp) const {
    // Generate unique key for experiment using precision level string
    return exp.experiment_id + "_" + precision_level_to_string(exp.config_used.precision_level);
}

std::string MetaQuantizationLearner::generate_hardware_key(const HardwareCapabilities& hw) const {
    // Generate key for hardware configuration
    return std::to_string(hw.gpu_memory_mb) + "_" +
           (hw.supports_fp16 ? "1" : "0") + "_" +
           (hw.supports_int8 ? "1" : "0") + "_" +
           (hw.supports_int4 ? "1" : "0");
}

QuantizationPerformance MetaQuantizationLearner::get_performance_from_experiment(const QuantizationExperiment& exp) const {
    // Extract performance metrics from experiment
    // Use actual experiment data if available, else estimate
    QuantizationPerformance perf;
    perf.precision = exp.config_used.precision_level;
    
    // Calculate actual execution time if available
    auto duration = exp.end_time - exp.start_time;
    perf.inference_speed_ms = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());
    
    // If no timing data, estimate based on precision level
    if (perf.inference_speed_ms <= 0.0f) {
        switch (exp.config_used.precision_level) {
            case PrecisionLevel::FP32:
                perf.inference_speed_ms = 15.0f;  // Baseline
                break;
            case PrecisionLevel::FP16:
                perf.inference_speed_ms = 10.0f;  // ~33% faster
                break;
            case PrecisionLevel::INT8:
                perf.inference_speed_ms = 7.0f;   // ~53% faster
                break;
            case PrecisionLevel::INT4:
                perf.inference_speed_ms = 5.0f;   // ~67% faster
                break;
            default:
                perf.inference_speed_ms = 15.0f;
                break;
        }
    }
    
    // Estimate memory usage based on original size and precision
    float memory_reduction_factor = 1.0f;
    switch (exp.config_used.precision_level) {
        case PrecisionLevel::FP32:
            memory_reduction_factor = 1.0f;
            break;
        case PrecisionLevel::FP16:
            memory_reduction_factor = 0.5f;
            break;
        case PrecisionLevel::INT8:
            memory_reduction_factor = 0.25f;
            break;
        case PrecisionLevel::INT4:
            memory_reduction_factor = 0.125f;
            break;
        default:
            memory_reduction_factor = 1.0f;
            break;
    }
    perf.memory_usage_mb = static_cast<uint32_t>((exp.original_model_size / (1024 * 1024)) * memory_reduction_factor);
    
    // Estimate accuracy based on quantization level
    // INT4/INT8 may have slight accuracy degradation vs FP32
    float accuracy_degradation = 0.0f;
    switch (exp.config_used.precision_level) {
        case PrecisionLevel::FP32:
            accuracy_degradation = 0.0f;
            break;
        case PrecisionLevel::FP16:
            accuracy_degradation = 0.01f;  // ~1% degradation
            break;
        case PrecisionLevel::INT8:
            accuracy_degradation = 0.03f;  // ~3% degradation
            break;
        case PrecisionLevel::INT4:
            accuracy_degradation = 0.08f;  // ~8% degradation
            break;
        default:
            accuracy_degradation = 0.0f;
            break;
    }
    perf.accuracy_score = 0.95f - accuracy_degradation;  // Baseline 95% accuracy
    
    // Use metadata if available
    if (exp.experiment_metadata.find("accuracy") != exp.experiment_metadata.end()) {
        perf.accuracy_score = exp.experiment_metadata.at("accuracy");
    }
    if (exp.experiment_metadata.find("memory_mb") != exp.experiment_metadata.end()) {
        perf.memory_usage_mb = static_cast<uint32_t>(exp.experiment_metadata.at("memory_mb"));
    }
    
    perf.measured_at = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Set hardware profile string (use static buffer since it's const char*)
    static thread_local std::string hw_profile_str;
    hw_profile_str = std::to_string(exp.hardware.gpu_memory_mb) + "MB_GPU";
    perf.hardware_profile = hw_profile_str.c_str();

    return perf;
}

// Evolutionary Quantization Optimizer implementation
EvolutionaryQuantizationOptimizer::EvolutionaryQuantizationOptimizer(std::shared_ptr<MetaQuantizationLearner> meta_learner)
    : meta_learner_(meta_learner), rng_(std::random_device{}()) {}

std::vector<QuantizationConfig> EvolutionaryQuantizationOptimizer::evolve_quantization_configs(
    const HardwareCapabilities& hw, const TaskRequirements& task,
    size_t population_size, size_t generations) {

    // Initialize population
    auto population = initialize_population(population_size, hw);

    for (size_t gen = 0; gen < generations; ++gen) {
        // Evaluate fitness
        std::vector<float> fitness_scores;
        for (const auto& config : population) {
            float fitness = meta_learner_->get_prediction_confidence(config);
            fitness_scores.push_back(fitness);
        }

        // Select best individuals
        auto selected = select_best(population, fitness_scores, population_size / 2);

        // Create new population through crossover and mutation
        std::vector<QuantizationConfig> new_population;
        new_population.insert(new_population.end(), selected.begin(), selected.end());

        // Fill rest through reproduction
        while (new_population.size() < population_size) {
            // Select parents
            size_t parent1_idx = std::uniform_int_distribution<size_t>(0, selected.size() - 1)(rng_);
            size_t parent2_idx = std::uniform_int_distribution<size_t>(0, selected.size() - 1)(rng_);

            // Crossover
            auto offspring = crossover_configs(selected[parent1_idx], selected[parent2_idx]);

            // Mutation
            offspring.first = mutate_config(offspring.first);
            offspring.second = mutate_config(offspring.second);

            new_population.push_back(offspring.first);
            if (new_population.size() < population_size) {
                new_population.push_back(offspring.second);
            }
        }

        population = new_population;
    }

    return population;
}

float EvolutionaryQuantizationOptimizer::evaluate_config_fitness(const QuantizationConfig& config,
                                                               const HardwareCapabilities& hw,
                                                               const QuantizationPerformance& performance) {
    // Evaluate fitness of quantization configuration
    float fitness = 0.0f;

    // Accuracy component (most important)
    fitness += performance.accuracy_score * 0.5f;

    // Speed component
    fitness += (100.0f / performance.inference_speed_ms) * 0.3f;

    // Memory efficiency component
    fitness += (1.0f - performance.memory_usage_mb / 8192.0f) * 0.2f;

    // Hardware compatibility bonus
    if ((config.precision_level == PrecisionLevel::FP16 && hw.supports_fp16) ||
        (config.precision_level == PrecisionLevel::INT8 && hw.supports_int8) ||
        (config.precision_level == PrecisionLevel::INT4 && hw.supports_int4)) {
        fitness += 0.1f;
    }

    return fitness;
}

QuantizationConfig EvolutionaryQuantizationOptimizer::mutate_config(const QuantizationConfig& config) {
    // Mutate quantization configuration
    QuantizationConfig mutated = config;

    // Random mutation with probability
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    if (dist(rng_) < mutation_rate_) {
        // Change precision level
        std::vector<PrecisionLevel> precisions = {PrecisionLevel::FP32, PrecisionLevel::FP16,
                                                PrecisionLevel::BF16, PrecisionLevel::INT8,
                                                PrecisionLevel::INT4};
        std::uniform_int_distribution<size_t> prec_dist(0, precisions.size() - 1);
        mutated.precision_level = precisions[prec_dist(rng_)];
    }

    return mutated;
}

std::pair<QuantizationConfig, QuantizationConfig> EvolutionaryQuantizationOptimizer::crossover_configs(
    const QuantizationConfig& parent1, const QuantizationConfig& parent2) {

    QuantizationConfig child1 = parent1;
    QuantizationConfig child2 = parent2;

    // Simple crossover - swap precision levels with probability
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    if (dist(rng_) < crossover_rate_) {
        child1.precision_level = parent2.precision_level;
        child2.precision_level = parent1.precision_level;
    }

    return {child1, child2};
}

std::vector<QuantizationConfig> EvolutionaryQuantizationOptimizer::initialize_population(
    size_t size, const HardwareCapabilities& hw) {

    std::vector<QuantizationConfig> population;

    std::vector<PrecisionLevel> available_precisions;
    available_precisions.push_back(PrecisionLevel::FP32);
    if (hw.supports_fp16) available_precisions.push_back(PrecisionLevel::FP16);
    if (hw.supports_fp16) available_precisions.push_back(PrecisionLevel::BF16); // Assume same as FP16
    if (hw.supports_int8) available_precisions.push_back(PrecisionLevel::INT8);
    if (hw.supports_int4) available_precisions.push_back(PrecisionLevel::INT4);

    std::uniform_int_distribution<size_t> dist(0, available_precisions.size() - 1);

    for (size_t i = 0; i < size; ++i) {
        QuantizationConfig config;
        config.precision_level = available_precisions[dist(rng_)];
        config.per_channel_quantization = true;
        config.calibration_factor = 1.0f;
        population.push_back(config);
    }

    return population;
}

std::vector<QuantizationConfig> EvolutionaryQuantizationOptimizer::select_best(
    const std::vector<QuantizationConfig>& population,
    const std::vector<float>& fitness_scores,
    size_t keep_count) {

    // Sort by fitness and keep top individuals
    std::vector<std::pair<float, QuantizationConfig>> scored;
    for (size_t i = 0; i < population.size(); ++i) {
        scored.push_back({fitness_scores[i], population[i]});
    }

    std::sort(scored.rbegin(), scored.rend()); // Sort descending by fitness

    std::vector<QuantizationConfig> selected;
    for (size_t i = 0; i < std::min(keep_count, scored.size()); ++i) {
        selected.push_back(scored[i].second);
    }

    return selected;
}

// Self-Evolving Quantization implementation
SelfEvolvingQuantization::SelfEvolvingQuantization(std::shared_ptr<AdaptiveQuantizationManager> quant_manager,
                                                 std::shared_ptr<MetaQuantizationLearner> meta_learner)
    : quantization_manager_(quant_manager), meta_learner_(meta_learner) {
    std::cout << "Self-Evolving Quantization initialized" << std::endl;
}

void SelfEvolvingQuantization::evolve_quantization_system() {
    // Main evolution cycle for quantization system
    std::cout << "Starting quantization system evolution cycle" << std::endl;

    // Analyze current performance
    EvolutionResults current_results = analyze_current_performance();

    // Generate improvements
    std::vector<CodeModification> improvements = generate_quantization_improvements();

    // Apply and validate improvements
    apply_self_modifications(current_results);

    std::cout << "Quantization system evolution cycle complete" << std::endl;
}

void SelfEvolvingQuantization::learn_from_system_performance(const SystemPerformanceMetrics& metrics) {
    // Learn from overall system performance
    // This would integrate with the system's performance monitoring

    std::cout << "Learning from system performance metrics" << std::endl;

    // Update evolution state based on metrics
    if (metrics.accuracy > 0.85f && metrics.memory_efficiency > 0.7f) {
        successful_evolution_cycles_++;
    }

    // Adapt evolution parameters
    if (evolution_history_.size() > 10) {
        adapt_evolution_parameters();
    }
}

void SelfEvolvingQuantization::apply_self_modifications(const EvolutionResults& results) {
    // Apply validated self-modifications
    std::cout << "Applying " << results.improvements.size() << " quantization improvements" << std::endl;

    // In practice, this would use the runtime compilation system to apply changes
    for (const auto& improvement : results.improvements) {
        learned_strategies_[improvement.improvement_type] = improvement.affected_config;
        std::cout << "Applied improvement: " << improvement.description << std::endl;
        // Push per-function strategies into the adaptive quantization manager so runtime can use them
        if (quantization_manager_) {
            quantization_manager_->apply_function_strategy(improvement.improvement_type, improvement.affected_config);
        }
    }

    evolution_history_.push_back(results);
}

void SelfEvolvingQuantization::meta_evolve_quantization_system() {
    // Meta-evolution of the quantization system itself
    std::cout << "Starting meta-evolution of quantization system" << std::endl;

    // Analyze evolution patterns
    auto patterns = detect_evolution_patterns();

    // Generate meta-improvements
    auto meta_improvements = generate_meta_improvements(patterns);

    // Apply meta-evolution
    for (const auto& improvement : meta_improvements) {
        apply_meta_evolution(improvement);
    }

    std::cout << "Meta-evolution complete" << std::endl;
}

// Private methods
EvolutionResults SelfEvolvingQuantization::analyze_current_performance() {
    // Analyze current quantization system performance
    EvolutionResults results;
    results.evolution_id = "quant_evolution_" + std::to_string(evolution_history_.size());
    results.timestamp = std::chrono::system_clock::now();

    // Simulate performance analysis
    QuantizationImprovement improvement;
    improvement.improvement_type = "precision_optimization";
    improvement.description = "Improved INT4 precision calibration";
    improvement.performance_gain = 0.02f; // 2% improvement
    improvement.affected_config.precision_level = PrecisionLevel::INT4;

    results.improvements.push_back(improvement);
    results.gains.accuracy_improvement = 0.02f;
    results.gains.speed_improvement = 0.05f;
    results.gains.memory_reduction = 50;

    return results;
}

std::vector<CodeModification> SelfEvolvingQuantization::generate_quantization_improvements() {
    // Generate potential improvements to quantization system
    std::vector<CodeModification> improvements;

    // Example improvement: better INT4 calibration
    CodeModification mod;
    mod.file_path = "src/vulkan/adaptive_quantization.cpp";
    mod.start_line = 200; // Approximate
    mod.end_line = 210;
    mod.original_code = "// Basic INT4 calibration";
    mod.new_code = "// Improved INT4 calibration with outlier handling";
    mod.reason = "Better handling of weight distribution outliers";

    improvements.push_back(mod);

    return improvements;
}

void SelfEvolvingQuantization::adapt_evolution_parameters() {
    // Adapt evolution parameters based on history
    if (successful_evolution_cycles_ > evolution_history_.size() * 0.7f) {
        // High success rate - be more aggressive
        evolution_aggressiveness_ *= 1.1f;
    } else {
        // Low success rate - be more conservative
        evolution_aggressiveness_ *= 0.9f;
    }
}

std::vector<std::string> SelfEvolvingQuantization::detect_evolution_patterns() {
    // Detect patterns in evolution history
    std::vector<std::string> patterns;

    if (evolution_history_.size() >= 3) {
        // Check for improvement trends
        float recent_gain = evolution_history_.back().gains.accuracy_improvement;
        float older_gain = evolution_history_[evolution_history_.size() - 3].gains.accuracy_improvement;

        if (recent_gain > older_gain * 1.5f) {
            patterns.push_back("accelerating_improvement");
        }
    }

    return patterns;
}

std::vector<std::string> SelfEvolvingQuantization::generate_meta_improvements(const std::vector<std::string>& patterns) {
    // Generate meta-level improvements
    std::vector<std::string> improvements;

    for (const auto& pattern : patterns) {
        if (pattern == "accelerating_improvement") {
            improvements.push_back("increase_evolution_frequency");
        }
    }

    return improvements;
}

void SelfEvolvingQuantization::apply_meta_evolution(const std::string& improvement) {
    // Apply meta-level evolution
    if (improvement == "increase_evolution_frequency") {
        evolution_frequency_ *= 1.2f;
        std::cout << "Increased evolution frequency for better adaptation" << std::endl;
    }
}

} // namespace Nyx