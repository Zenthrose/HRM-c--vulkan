#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include "../hrm/hrm.hpp"
#include "../utils/utf8_processor.hpp"
#include "../hrm/meta_reasoning_layer.hpp"

struct SelfEvolvingHRMConfig {
    HRMConfig hrm_config;
    UTF8Config utf8_config;
    MetaReasoningConfig meta_config;

    // Self-evolution parameters
    bool enable_self_evolution;
    float evolution_rate;
    int adaptation_cycles;
    bool enable_continual_learning;

    // Communication parameters
    bool use_utf8_communication;
    int max_conversation_length;
};

struct CommunicationResult {
    std::string response;
    float confidence_score;
    bool self_repair_performed;
    int evolution_cycles_completed;
    std::vector<std::string> detected_issues;
    std::vector<std::string> applied_corrections;
};

class SelfEvolvingHRM {
public:
    SelfEvolvingHRM(const SelfEvolvingHRMConfig& config);
    ~SelfEvolvingHRM() = default;

    // Core communication interface
    CommunicationResult communicate(const std::string& input_message);

    // Text generation (virtual for subclasses to override)
    virtual std::string generate_text(const std::string& prompt, uint32_t max_length = 100);

    // Self-evolution methods
    void perform_evolution_cycle();
    void adapt_to_feedback(const std::string& feedback);
    void learn_from_interaction(const std::string& input, const std::string& response,
                               const CommunicationResult& result);

    // Self-repair interface
    CommunicationResult repair_and_respond(const std::string& input_message,
                                          const std::string& flawed_response);

    // System status
    void get_system_status(std::unordered_map<std::string, std::string>& status);

    // Access to core HRM model
    HRM* get_hrm() { return hrm_model_.get(); }

    // Get configuration
    const SelfEvolvingHRMConfig& get_config() const { return config_; }

protected:
    SelfEvolvingHRMConfig config_;
    std::unique_ptr<HRM> hrm_model_;
    std::unique_ptr<UTF8Processor> utf8_processor_;
    std::unique_ptr<MetaReasoningLayer> meta_reasoner_;

    // Evolution state
    int evolution_cycles_completed_;
    float average_confidence_;
    int total_interactions_;
    std::unordered_map<std::string, float> learned_patterns_;

    // Internal processing methods
    std::string process_input(const std::string& input);
    std::string generate_response(const std::string& processed_input, HRMCarry& carry);
    CommunicationResult analyze_and_improve_response(const std::string& input,
                                                   const std::string& raw_response,
                                                   const HRMCarry& carry);

    // Evolution mechanisms
    void update_internal_parameters();
    void evolve_architecture();
    void consolidate_learned_patterns();

    // Evolution helper methods
    float calculate_system_efficiency();
    float assess_adaptation_potential();

    // Conversation memory
    std::vector<std::pair<std::string, std::string>> conversation_history_;
    void maintain_conversation_context();
};