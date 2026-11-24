#include "ops/moe/routed_swiglu/cuda/dispatch.h"

namespace ninfer::ops::detail::moe_routed {
namespace {
constexpr int kOrderedExperts = 16, kWideAverage = 128;
constexpr int kWideStart =
    (kOrderedExperts + 1) * kWideAverage / kernel::block_scaled::clamped::kTopK + 1;

Plan select_plan(const SwiGLUProfile& ffn, int t, int) {
    Plan plan;
    plan.implementation = Implementation::ClampedBlockScaled;
    const int c         = t <= 8 ? 8 : t <= 16 ? 16 : t <= 32 ? 32 : 64;
    plan.clamped        = {.format                  = ffn.gate_up_type,
                           .gate_columns            = t >= 257 ? 128 : c,
                           .down_columns            = c,
                           .direct_assignments      = t <= 2,
                           .atomic_output           = t >= 3,
                           .single_block_grouping   = t <= 64,
                           .token_ordered_groups    = true,
                           .queued_ctas_per_sm      = 6,
                           .vector_gate_ctas_per_sm = t == 2 ? 8 : 0,
                           .clamp_limit             = *ffn.clamp_limit};
    if (t >= 3)
        plan.clamped.fp8 = {.row_window           = 4,
                            .ordered_expert_limit = kOrderedExperts,
                            .wide_min_average     = kWideAverage,
                            .wide_down            = t >= kWideStart,
                            .dense_down           = t >= 1024};
    return plan;
}
} // namespace

const ShapePolicy& fp8_h4096_i2048_e288_k8() {
    static constexpr int boundaries[]{2, 3, 9, 17, 33, 65, 257, kWideStart, 1024};
    static const ShapePolicy policy{.select                = select_plan,
                                    .allocation_boundaries = boundaries,
                                    .scaled_shared         = false,
                                    .accumulate            = false};
    return policy;
}
} // namespace ninfer::ops::detail::moe_routed
