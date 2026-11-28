#pragma once

#include "ninfer/ops/moe.h"
#include "ops/op_check.h"

namespace ninfer::test {

// The oracle uses the public inputs and decoded weights through both projections.
inline ReductionCriterion block_scaled_moe_criterion(const ops::SwiGLUProfile& ffn) {
    if (ops::allows_a4(ffn.gate_up_policy) || ops::allows_a4(ffn.down_policy))
        return {.25, 1e-7, .35};
    if (ops::allows_a8(ffn.gate_up_policy) || ops::allows_a8(ffn.down_policy))
        return {.08, 1e-7, .12};
    return {.012, 1e-6, .008};
}

} // namespace ninfer::test
