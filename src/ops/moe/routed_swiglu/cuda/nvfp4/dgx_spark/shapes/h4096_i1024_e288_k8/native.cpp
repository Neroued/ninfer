#include "ops/moe/routed_swiglu/cuda/dispatch.h"

namespace ninfer::ops::detail::moe_routed {
namespace {
Plan select_native(const SwiGLUProfile& ffn, int tokens, int) {
    using namespace moe_cuda::block_scaled;
    clamped::Plan p;
    p.format                  = ffn.gate_up_type;
    p.arithmetic              = allows_a4(ffn.gate_up_policy) ? clamped::Nvfp4Arithmetic::MultiplyA4
                                                              : clamped::Nvfp4Arithmetic::MultiplyA16;
    p.intermediate            = ffn.intermediate;
    p.native_a4_grouped_input = p.arithmetic == clamped::Nvfp4Arithmetic::MultiplyA4 &&
                                (tokens > 2 || ffn.gate_up_layout != QuantLayout::Csf32);
    if (p.arithmetic == clamped::Nvfp4Arithmetic::MultiplyA4 &&
        ffn.gate_up_layout == QuantLayout::Csf32)
        p.native_a4_vector_max_load = 2;
    p.gate_columns = p.down_columns = tokens <= 8 ? 8 : 16;
    p.atomic_output                 = true;
    p.single_block_grouping         = tokens <= (allows_a4(ffn.gate_up_policy) ? 256 : 64);
    p.token_ordered_groups          = true;
    p.clamp_limit                   = *ffn.clamp_limit;
    p.nvfp4                         = {Nvfp4Pipeline::K128, Nvfp4CachePolicy::Cached, 0};
    if (tokens > 16 && p.arithmetic == clamped::Nvfp4Arithmetic::MultiplyA4) {
        p.gate_columns = p.down_columns = 64;
        p.small_expert_limit            = 64;
        p.group_columns                 = 128;
        p.dense = {.gate_columns = 128, .down_columns = 128, .queued_ctas_per_sm = 2};
    }
    if (p.arithmetic == clamped::Nvfp4Arithmetic::MultiplyA4 && tokens >= 128)
        p.dense_down_columns = 128;
    if (p.arithmetic == clamped::Nvfp4Arithmetic::MultiplyA4 && tokens >= 2048) {
        p.pool_expert_limit       = 16;
        p.pool_columns            = 128;
        p.pool_row_window         = 4;
        p.pool_queued_ctas_per_sm = 3;
    }
    if (p.native_a4_grouped_input) {
        p.group_columns      = 256;
        p.column_major_down  = true;
        p.dense_down_columns = 0;
        p.pool_expert_limit  = 0;
    }
    if (p.arithmetic == clamped::Nvfp4Arithmetic::MultiplyA16) {
        p.gate_columns = p.down_columns = 8;
        p.group_columns                 = 128;
        p.a16.vector_max_load           = 2;
        p.a16.buckets[0]                = {8, 8, 8,
                            tokens <= 8 ? clamped::Nvfp4A16Tile{64, 64, 128, 2}
                                                       : clamped::Nvfp4A16Tile{128, 128, 128, 1}};
        p.a16.buckets[1]                = {32, 32, 32, {128, 128, 128, 1}};
        p.a16.buckets[2]                = {128, 64, 64, {128, 128, 128, 1}};
        p.a16.fp16_product = ffn.gate_up_layout != QuantLayout::Csf32 && *ffn.clamp_limit == 10.F;
        if (p.a16.fp16_product) {
            p.nvfp4.down_row_window = 4;
            p.column_major_down     = !p.single_block_grouping;
            p.a16.buckets[3]        = {192, 128, 128, {128, 128, 64, 2}};
            p.a16.buckets[4]        = {INT_MAX, 64, 64, {128, 128, 64, 2}};
        } else {
            p.a16.buckets[3] = {INT_MAX, 64, 64, {128, 128, 64, 2}};
        }
    }
    Plan plan;
    plan.implementation = Implementation::ClampedBlockScaled;
    plan.clamped        = p;
    return plan;
}
} // namespace

const ShapePolicy& nvfp4_mul_h4096_i1024_e288_k8() {
    static constexpr int boundaries[]{3, 9, 17, 65, 128, 257, 513};
    static const ShapePolicy policy{.select                = select_native,
                                    .allocation_boundaries = boundaries,
                                    .all_slice_tokens      = 4096,
                                    .masked_slice_tokens   = 4096,
                                    .scaled_shared         = false,
                                    .accumulate            = true};
    return policy;
}
} // namespace ninfer::ops::detail::moe_routed
