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

private:
    HRMConfig config_;
    std::unique_ptr<HRMInner> inner_;
};