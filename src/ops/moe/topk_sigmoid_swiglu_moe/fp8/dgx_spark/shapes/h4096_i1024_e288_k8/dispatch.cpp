#include "ops/moe/topk_sigmoid_swiglu_moe/dispatch.h"

namespace ninfer::ops::detail::moe_sigmoid {
namespace {
moe_cuda::block_scaled::clamped::Plan select_plan(const SwiGLUProfile& ffn, int t) {
    moe_cuda::block_scaled::clamped::Plan p{.format                  = ffn.gate_up_type,
                                            .intermediate            = ffn.intermediate,
                                            .gate_columns            = t <= 8     ? 8
                                                                       : t <= 32  ? 16
                                                                       : t <= 64  ? 32
                                                                       : t <= 128 ? 16
                                                                                  : 32,
                                            .down_columns            = t <= 8     ? 8
                                                                       : t <= 32  ? 16
                                                                       : t <= 64  ? 32
                                                                       : t <= 128 ? 16
                                                                                  : 32,
                                            .direct_assignments      = t <= 2,
                                            .atomic_output           = t >= 3,
                                            .single_block_grouping   = t <= 64,
                                            .token_ordered_groups    = true,
                                            .queued_ctas_per_sm      = t > 32 && t <= 128 ? 12 : 6,
                                            .vector_gate_ctas_per_sm = t == 2 ? 8 : 0,
                                            .clamp_limit             = *ffn.clamp_limit};
    if (t >= 3)
        p.fp8 = {.row_window           = 4,
                 .ordered_expert_limit = 16,
                 .dense_down           = t >= 64,
                 .k_step               = t <= 64 ? 256 : 0,
                 .k256_max_load        = 32,
                 .large_gate_columns   = 128,
                 .large_down_columns   = 64};
    return p;
}
} // namespace

const ShapePolicy& fp8_h4096_i1024_e288_k8() {
    static constexpr int boundaries[]{2, 3, 9, 33, 64, 65, 128, 129};
    static const ShapePolicy policy{select_plan, boundaries, 4096};
    return policy;
}
} // namespace ninfer::ops::detail::moe_sigmoid
