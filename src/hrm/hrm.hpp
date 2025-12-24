#pragma once

#include <unordered_map>
#include <vector>
#include "hrm_inner.hpp"

struct HRMCarry {
    HRMInnerCarry inner_carry;
    std::vector<int32_t> steps;
    std::vector<bool> halted;
    std::unordered_map<std::string, Tensor> current_data;
};

struct HRMConfig {
    HRMInnerConfig inner_config;
    bool is_training = false;
};

class HRM {
public:
    HRM(const HRMConfig& config);
    ~HRM() = default;

    HRMCarry initial_carry(const std::unordered_map<std::string, Tensor>& batch);
    std::pair<HRMCarry, std::unordered_map<std::string, Tensor>> forward(
        const HRMCarry& carry,
        const std::unordered_map<std::string, Tensor>& batch
    );

    std::unordered_map<std::string, Tensor> backward(
        const HRMCarry& carry,
        const std::unordered_map<std::string, Tensor>& batch,
        const std::unordered_map<std::string, Tensor>& output_grads
    );

    void set_training_mode(bool training) { config_.is_training = training; }
    bool is_training() const { return config_.is_training; }

    // Access inner model for parameter inspection and updates
    HRMInner* get_inner() { return inner_.get(); }

private:
    HRMConfig config_;
    std::unique_ptr<HRMInner> inner_;
};