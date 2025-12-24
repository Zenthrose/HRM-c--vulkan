#pragma once

#include <vector>
#include <cstdint>

namespace Nyx {

struct TrainingBatch {
    std::vector<float> input_sequences;
    std::vector<uint32_t> target_sequences;
    uint32_t batch_size = 0;
    uint32_t seq_length = 0;
};

} // namespace Nyx

// Backwards-compatibility: allow unqualified `TrainingBatch` name in global namespace
using Nyx::TrainingBatch;
