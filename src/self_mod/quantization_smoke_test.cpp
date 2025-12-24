#include "../vulkan/adaptive_quantization.hpp"
#include <cassert>
#include <iostream>

int main() {
    // Create manager with no engine (we only test strategy storage APIs)
    Nyx::AdaptiveQuantizationManager mgr(nullptr);

    Nyx::QuantizationConfig cfg;
    cfg.precision_level = Nyx::PrecisionLevel::INT8;
    cfg.per_channel_quantization = false;
    cfg.calibration_factor = 1.0f;

    std::string key = "test_model:funcA";
    mgr.apply_function_strategy(key, cfg);

    Nyx::QuantizationConfig out;
    bool ok = mgr.get_function_strategy(key, out);
    if (!ok) {
        std::cerr << "Failed to retrieve stored function strategy" << std::endl;
        return 2;
    }
    assert(ok);
    assert(out.precision_level == cfg.precision_level);

    std::cout << "Quantization strategy storage smoke test passed" << std::endl;
    return 0;
}
