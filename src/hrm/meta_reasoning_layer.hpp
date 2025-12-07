#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include "../core/attention.hpp" // For Tensor
#include "hrm.hpp" // For HRM model access

struct MetaReasoningConfig {
    int analysis_depth;
    float confidence_threshold;
    int max_correction_attempts;
    bool enable_self_repair;
    bool enable_confidence_scoring;

    // Model access for meta-analysis
    HRM* hrm_model;
};

struct AnalysisResult {
    float confidence_score;
    std::vector<std::string> detected_issues;
    std::vector<std::string> suggested_corrections;
    bool requires_repair;
};

struct RepairResult {
    bool repair_successful;
    std::string repaired_output;
    float improvement_score;
    int attempts_used;
};

class MetaReasoningLayer {
public:
    MetaReasoningLayer(const MetaReasoningConfig& config);
    ~MetaReasoningLayer() = default;

    // Core meta-reasoning functions
    AnalysisResult analyze_output(const std::string& input, const std::string& output,
                                 const HRMCarry& model_state);

    RepairResult attempt_repair(const std::string& input, const std::string& flawed_output,
                               const AnalysisResult& analysis, const HRMCarry& model_state);

    // Confidence and consistency checking
    float compute_output_confidence(const Tensor& logits, const std::string& output);
    bool check_logical_consistency(const std::string& input, const std::string& output);
    std::vector<std::string> detect_contradictions(const std::string& text);

    // Self-improvement functions
    void learn_from_correction(const std::string& original_output,
                              const std::string& corrected_output,
                              const RepairResult& repair_result);

private:
    MetaReasoningConfig config_;

    // Internal analysis functions
    std::vector<std::string> analyze_semantic_coherence(const std::string& text);
    std::vector<std::string> analyze_syntactic_correctness(const std::string& text);
    std::vector<std::string> analyze_logical_soundness(const std::string& input, const std::string& output);

    // Repair strategies
    std::string apply_local_corrections(const std::string& text, const std::vector<std::string>& issues);
    std::string regenerate_problematic_sections(const std::string& text,
                                               const std::vector<size_t>& problem_positions);
    std::string combine_multiple_corrections(const std::vector<std::string>& correction_candidates);

    // Learning and adaptation
    void update_correction_patterns(const std::string& pattern, const std::string& correction, float success_rate);
    std::unordered_map<std::string, std::pair<std::string, float>> correction_patterns_;

    // Helper functions
    std::vector<std::string> tokenize_text(const std::string& text);
    float calculate_semantic_similarity(const std::string& text1, const std::string& text2);
    bool is_grammatically_correct(const std::string& sentence);
};