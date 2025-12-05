#include "self_evolving_hrm.hpp"
#include <iostream>
#include <algorithm>
#include <random>

SelfEvolvingHRM::SelfEvolvingHRM(const SelfEvolvingHRMConfig& config) : config_(config) {
    std::cout << "Initializing Self-Evolving HRM System..." << std::endl;

    // Initialize core components
    std::cout << "Creating HRM model..." << std::endl;
    hrm_model_ = std::make_unique<HRM>(config.hrm_config);
    std::cout << "HRM model created successfully" << std::endl;

    if (config.use_utf8_communication) {
        std::cout << "Creating UTF-8 processor..." << std::endl;
        utf8_processor_ = std::make_unique<UTF8Processor>(config.utf8_config);
        std::cout << "UTF-8 processor created successfully" << std::endl;
    }

    // Set up meta-reasoning with reference to HRM model
    std::cout << "Creating meta-reasoning layer..." << std::endl;
    MetaReasoningConfig meta_config = config.meta_config;
    meta_config.hrm_model = hrm_model_.get();
    meta_reasoner_ = std::make_unique<MetaReasoningLayer>(meta_config);
    std::cout << "Meta-reasoning layer created successfully" << std::endl;

    // Initialize evolution state
    evolution_cycles_completed_ = 0;
    average_confidence_ = 0.5f;
    total_interactions_ = 0;

    std::cout << "Self-Evolving HRM System initialized with "
              << (config.enable_self_evolution ? "self-evolution enabled" : "self-evolution disabled")
              << " and "
              << (config.use_utf8_communication ? "UTF-8 communication" : "standard communication")
              << std::endl;
}

CommunicationResult SelfEvolvingHRM::communicate(const std::string& input_message) {
    std::cout << "Processing communication: " << input_message.substr(0, 50) << "..." << std::endl;

    // Process input through UTF-8 processor if enabled
    std::string processed_input = config_.use_utf8_communication ?
        utf8_processor_->decode_utf8(utf8_processor_->encode_utf8(input_message)) : input_message;

    // Get initial carry state
    std::unordered_map<std::string, Tensor> batch;
    // Simplified batch creation - in practice this would be more sophisticated
    batch["inputs"] = Tensor(); // Placeholder
    batch["puzzle_identifiers"] = Tensor(); // Placeholder

    HRMCarry carry = hrm_model_->initial_carry(batch);

    // Generate initial response
    std::string raw_response = generate_response(processed_input, carry);

    // Analyze and potentially repair the response
    CommunicationResult result = analyze_and_improve_response(processed_input, raw_response, carry);

    // Learn from this interaction
    if (config_.enable_continual_learning) {
        learn_from_interaction(input_message, result.response, result);
    }

    // Perform evolution if enabled and conditions met
    if (config_.enable_self_evolution && total_interactions_ % config_.adaptation_cycles == 0) {
        perform_evolution_cycle();
    }

    // Maintain conversation context
    maintain_conversation_context();
    conversation_history_.push_back({input_message, result.response});

    total_interactions_++;

    return result;
}

CommunicationResult SelfEvolvingHRM::analyze_and_improve_response(const std::string& input,
                                                               const std::string& raw_response,
                                                               const HRMCarry& carry) {
    CommunicationResult result;
    result.response = raw_response;
    result.self_repair_performed = false;
    result.evolution_cycles_completed = evolution_cycles_completed_;

    // Analyze the response using meta-reasoning
    AnalysisResult analysis = meta_reasoner_->analyze_output(input, raw_response, carry);

    result.confidence_score = analysis.confidence_score;
    result.detected_issues = analysis.detected_issues;

    // Attempt repair if needed
    if (analysis.requires_repair) {
        RepairResult repair = meta_reasoner_->attempt_repair(input, raw_response, analysis, carry);

        if (repair.repair_successful) {
            result.response = repair.repaired_output;
            result.self_repair_performed = true;
            result.applied_corrections.push_back("Self-repair applied: " +
                                               std::to_string(repair.improvement_score) + " improvement");

            // Learn from successful repair
            meta_reasoner_->learn_from_correction(raw_response, repair.repaired_output, repair);
        }
    }

    // Update confidence tracking
    average_confidence_ = (average_confidence_ * total_interactions_ + result.confidence_score) /
                         (total_interactions_ + 1);

    return result;
}

void SelfEvolvingHRM::perform_evolution_cycle() {
    std::cout << "Performing evolution cycle #" << evolution_cycles_completed_ + 1 << std::endl;

    update_internal_parameters();
    evolve_architecture();
    consolidate_learned_patterns();

    evolution_cycles_completed_++;

    std::cout << "Evolution cycle completed. Current confidence: " << average_confidence_ << std::endl;
}

void SelfEvolvingHRM::adapt_to_feedback(const std::string& feedback) {
    // Process feedback and adapt system parameters
    std::cout << "Adapting to feedback: " << feedback << std::endl;

    // Simple adaptation based on feedback keywords
    if (feedback.find("too verbose") != std::string::npos) {
        // Reduce response length
        std::cout << "Adapting: Reducing verbosity" << std::endl;
    } else if (feedback.find("too brief") != std::string::npos) {
        // Increase response detail
        std::cout << "Adapting: Increasing detail" << std::endl;
    } else if (feedback.find("inaccurate") != std::string::npos) {
        // Improve accuracy mechanisms
        std::cout << "Adapting: Enhancing accuracy checks" << std::endl;
    }

    // Trigger evolution cycle
    perform_evolution_cycle();
}

void SelfEvolvingHRM::learn_from_interaction(const std::string& input, const std::string& response,
                                          const CommunicationResult& result) {
    // Extract patterns from successful interactions
    if (result.confidence_score > 0.8f) {
        // Learn successful patterns
        std::string pattern_key = input.substr(0, std::min(size_t(20), input.length()));
        learned_patterns_[pattern_key] = std::max(learned_patterns_[pattern_key], result.confidence_score);
    }
}

void SelfEvolvingHRM::get_system_status(std::unordered_map<std::string, std::string>& status) {
    status["evolution_cycles"] = std::to_string(evolution_cycles_completed_);
    status["average_confidence"] = std::to_string(average_confidence_);
    status["total_interactions"] = std::to_string(total_interactions_);
    status["learned_patterns"] = std::to_string(learned_patterns_.size());
    status["conversation_length"] = std::to_string(conversation_history_.size());
    status["self_evolution_enabled"] = config_.enable_self_evolution ? "true" : "false";
    status["utf8_communication"] = config_.use_utf8_communication ? "true" : "false";
    status["continual_learning"] = config_.enable_continual_learning ? "true" : "false";
}

CommunicationResult SelfEvolvingHRM::repair_and_respond(const std::string& input_message,
                                                     const std::string& flawed_response) {
    // Force repair of a known flawed response
    std::unordered_map<std::string, Tensor> batch;
    HRMCarry carry = hrm_model_->initial_carry(batch);

    AnalysisResult analysis = meta_reasoner_->analyze_output(input_message, flawed_response, carry);
    RepairResult repair = meta_reasoner_->attempt_repair(input_message, flawed_response, analysis, carry);

    CommunicationResult result;
    result.response = repair.repair_successful ? repair.repaired_output : flawed_response;
    result.confidence_score = repair.repair_successful ? analysis.confidence_score + repair.improvement_score : analysis.confidence_score;
    result.self_repair_performed = repair.repair_successful;
    result.detected_issues = analysis.detected_issues;
    result.applied_corrections.push_back("Forced repair: " + std::to_string(repair.attempts_used) + " attempts");

    return result;
}

// Private methods

std::string SelfEvolvingHRM::process_input(const std::string& input) {
    // Apply any preprocessing based on learned patterns
    std::string processed = input;

    // Use UTF-8 processing if enabled
    if (config_.use_utf8_communication && utf8_processor_) {
        // Validate UTF-8
        if (!utf8_processor_->is_valid_utf8(input)) {
            std::cout << "Warning: Invalid UTF-8 detected in input" << std::endl;
            auto invalid_positions = utf8_processor_->find_invalid_sequences(input);
            std::cout << "Invalid sequences at positions: ";
            for (size_t pos : invalid_positions) {
                std::cout << pos << " ";
            }
            std::cout << std::endl;
        }
    }

    return processed;
}

std::string SelfEvolvingHRM::generate_response(const std::string& processed_input, HRMCarry& carry) {
    // TODO: Implement actual AI-generated response using trained language model
    // This should use the HRM's forward pass for text generation with proper sampling
    // For now, provide a basic response that acknowledges the input without repetition

    // Store input in conversation history for context
    conversation_history_.push_back({"User", processed_input});

    // Limit history to prevent unbounded growth
    if (conversation_history_.size() > config_.max_conversation_length) {
        conversation_history_.erase(conversation_history_.begin());
    }

    // Generate response using HRM model (placeholder - needs implementation)
    // In full implementation: tokenize input, run through HRM layers, sample output tokens
    std::string response = "I acknowledge your message about: " + processed_input.substr(0, 30) + "... ";

    // Add contextual element based on conversation history
    if (conversation_history_.size() > 1) {
        response += "Building on our previous discussion, ";
    }

    response += "my reasoning system is processing this information. What aspect would you like me to focus on?";

    conversation_history_.push_back({"HRM", response});

    return response;
}

void SelfEvolvingHRM::update_internal_parameters() {
    // Adapt system parameters based on performance
    if (average_confidence_ < 0.7f) {
        // Increase analysis depth
        std::cout << "Evolution: Increasing analysis depth due to low confidence" << std::endl;
    } else if (average_confidence_ > 0.9f) {
        // Optimize for efficiency
        std::cout << "Evolution: Optimizing for efficiency due to high confidence" << std::endl;
    }
}

void SelfEvolvingHRM::evolve_architecture() {
    // Placeholder for architecture evolution
    // In a full implementation, this could modify layer sizes, add new components, etc.
    std::cout << "Evolution: Architecture adaptation (placeholder)" << std::endl;
}

void SelfEvolvingHRM::consolidate_learned_patterns() {
    // Clean up and consolidate learned patterns
    if (learned_patterns_.size() > 1000) {
        // Remove least successful patterns
        std::vector<std::pair<std::string, float>> patterns(learned_patterns_.begin(), learned_patterns_.end());
        std::sort(patterns.begin(), patterns.end(),
                 [](const auto& a, const auto& b) { return a.second < b.second; });

        // Keep only top 80%
        size_t keep_count = patterns.size() * 4 / 5;
        learned_patterns_.clear();
        for (size_t i = patterns.size() - keep_count; i < patterns.size(); ++i) {
            learned_patterns_[patterns[i].first] = patterns[i].second;
        }

        std::cout << "Evolution: Consolidated patterns, kept " << learned_patterns_.size() << " most successful" << std::endl;
    }
}

void SelfEvolvingHRM::maintain_conversation_context() {
    // Limit conversation history to prevent unbounded growth
    if (conversation_history_.size() > config_.max_conversation_length) {
        size_t remove_count = conversation_history_.size() - config_.max_conversation_length;
        conversation_history_.erase(conversation_history_.begin(),
                                   conversation_history_.begin() + remove_count);
    }
}