#pragma once

#include "ninfer/ops/moe.h"

namespace ninfer::ops::detail::moe_cpu {

struct Invocation {
    const std::uint16_t* input;
    const std::int32_t* ids;
    const float* coefficients;
    const CpuSwiGLUExperts& experts;
    ExpertSubsetKind subset;
    const std::int32_t* mask;
    float* output;
    int tokens;
};

inline bool participates(ExpertSubsetKind kind, const std::int32_t* mask, int expert) noexcept {
    switch (kind) {
    case ExpertSubsetKind::All:
        return true;
    case ExpertSubsetKind::None:
        return false;
    case ExpertSubsetKind::MaskOnes:
        return mask[expert] == 1;
    case ExpertSubsetKind::MaskZeros:
        return mask[expert] == 0;
    }
    return false;
}

} // namespace ninfer::ops::detail::moe_cpu
