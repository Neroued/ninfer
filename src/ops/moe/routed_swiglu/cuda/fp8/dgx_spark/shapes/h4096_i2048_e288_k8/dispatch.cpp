#include "ops/moe/routed_swiglu/cuda/dispatch.h"

namespace ninfer::ops::detail::moe_routed {
namespace {
Plan select_plan(const SwiGLUProfile& ffn, int t, int) {
    Plan plan;
    plan.implementation = Implementation::ClampedBlockScaled;
    const int c         = t <= 16 ? 8 : t <= 32 ? 16 : t <= 64 ? 32 : 64;
    plan.clamped        = {.format                 = ffn.gate_up_type,
                           .gate_columns           = t >= 257 ? 128 : c,
                           .down_columns           = c,
                           .vector                 = t <= 2,
                           .atomic_output          = t >= 128,
                           .single_block_grouping  = t <= 64,
                           .column_major_down      = false,
                           .preserve_balanced_down = true,
                           .queued_ctas_per_sm     = 6,
                           .clamp_limit            = *ffn.clamp_limit};
    return plan;
}
} // namespace

const ShapePolicy& fp8_h4096_i2048_e288_k8() {
    static constexpr int boundaries[]{2, 3, 17, 33, 65, 128, 129, 257, 512, 1024};
    static const ShapePolicy policy{.select                = select_plan,
                                    .allocation_boundaries = boundaries,
                                    .scaled_shared         = false,
                                    .accumulate            = false};
    return policy;
}
} // namespace ninfer::ops::detail::moe_routed
