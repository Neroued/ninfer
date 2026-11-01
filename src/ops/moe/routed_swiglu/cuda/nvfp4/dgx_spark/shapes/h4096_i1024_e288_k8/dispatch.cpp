#include "ops/moe/routed_swiglu/cuda/dispatch.h"

namespace ninfer::ops::detail::moe_routed {
namespace {
Plan select_plan(const SwiGLUProfile& ffn, int tokens, int) {
    using moe_cuda::block_scaled::Nvfp4CachePolicy;
    using moe_cuda::block_scaled::Nvfp4Pipeline;
    moe_cuda::block_scaled::clamped::Plan p{.format                 = ffn.gate_up_type,
                                            .intermediate           = ffn.intermediate,
                                            .gate_columns           = 8,
                                            .down_columns           = 8,
                                            .dense_down_columns     = tokens >= 128 ? 128 : 0,
                                            .direct_assignments     = tokens == 1,
                                            .atomic_output          = true,
                                            .single_block_grouping  = tokens <= 64,
                                            .column_major_down      = tokens >= 1024,
                                            .preserve_balanced_down = true,
                                            .token_ordered_groups   = true,
                                            .queued_ctas_per_sm     = 6,
                                            .clamp_limit            = *ffn.clamp_limit};
    p.nvfp4 = {Nvfp4Pipeline::K256Deep, Nvfp4CachePolicy::BypassWeightL1, 4};
    if (tokens > 8 && tokens <= 16) p.gate_columns = p.down_columns = 16;
    if (tokens > 16) {
        p.small_expert_limit = 8;
        p.medium             = {.gate_columns       = tokens > 2048   ? 128
                                                      : tokens > 1024 ? 64
                                                                      : 32,
                                .down_columns       = tokens > 2048   ? 128
                                                      : tokens > 1024 ? 64
                                                                      : 32,
                                .max_load           = tokens > 1024 ? 128 : 64,
                                .tuning             = {Nvfp4Pipeline::K256, Nvfp4CachePolicy::RetainInputs, 4},
                                .queued_ctas_per_sm = 6};
        p.dense              = {.gate_columns       = 128,
                                .down_columns       = 128,
                                .tuning             = {Nvfp4Pipeline::K128, Nvfp4CachePolicy::Cached, 0},
                                .queued_ctas_per_sm = 2};
        p.group_columns      = 128;
    }
    if (tokens >= 1024) {
        p.pool_min_experts        = 8;
        p.pool_expert_limit       = 16;
        p.pool_columns            = 128;
        p.pool_row_window         = 4;
        p.pool_queued_ctas_per_sm = 3;
    }
    Plan plan;
    plan.implementation = Implementation::ClampedBlockScaled;
    plan.clamped        = p;
    return plan;
}
} // namespace

const ShapePolicy& nvfp4_h4096_i1024_e288_k8() {
    static constexpr int boundaries[]{2, 9, 17, 65, 128, 1024, 1025, 2049};
    static const ShapePolicy policy{.select                = select_plan,
                                    .allocation_boundaries = boundaries,
                                    .all_slice_tokens      = 4096,
                                    .masked_slice_tokens   = 4096,
                                    .scaled_shared         = false,
                                    .accumulate            = false};
    return policy;
}
} // namespace ninfer::ops::detail::moe_routed
