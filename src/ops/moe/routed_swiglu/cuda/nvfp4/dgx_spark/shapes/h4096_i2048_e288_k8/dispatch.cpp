#include "ops/moe/routed_swiglu/cuda/dispatch.h"

namespace ninfer::ops::detail::moe_routed {
namespace {
Plan select_plan(const SwiGLUProfile& ffn, int t, int) {
    Plan plan;
    plan.implementation = Implementation::ClampedBlockScaled;
    const int columns   = t <= 16 ? 8 : t <= 128 ? 32 : t < 512 ? 64 : 128;
    plan.clamped        = {.format                 = ffn.gate_up_type,
                           .gate_columns           = columns,
                           .down_columns           = columns,
                           .vector                 = t == 1,
                           .atomic_output          = t >= 128,
                           .single_block_grouping  = t <= 64,
                           .column_major_down      = t >= 1024,
                           .preserve_balanced_down = true,
                           .queued_ctas_per_sm     = 6,
                           .clamp_limit            = *ffn.clamp_limit};
    return plan;
}
} // namespace

const ShapePolicy& nvfp4_h4096_i2048_e288_k8() {
    static constexpr int boundaries[]{2, 3, 17, 33, 65, 128, 129, 257, 512, 1024};
    static const ShapePolicy policy{.select                = select_plan,
                                    .allocation_boundaries = boundaries,
                                    .scaled_shared         = false,
                                    .accumulate            = false};
    return policy;
}
} // namespace ninfer::ops::detail::moe_routed
