#include "ops/moe/routed_swiglu/cuda/dispatch.h"

#include <algorithm>

namespace ninfer::ops::detail::moe_routed {
namespace {
Plan select_plan(const SwiGLUProfile& ffn, int t, int) {
    Plan plan;
    plan.implementation = Implementation::ClampedBlockScaled;
    auto& p             = plan.clamped;
    p                   = {.format                 = ffn.gate_up_type,
                           .gate_columns           = 8,
                           .down_columns           = 8,
                           .vector                 = t == 1,
                           .atomic_output          = t > 64,
                           .single_block_grouping  = t <= 64,
                           .column_major_down      = t >= 1024,
                           .preserve_balanced_down = true,
                           .queued_ctas_per_sm     = 6,
                           .clamp_limit            = *ffn.clamp_limit};
    using kernel::block_scaled::Nvfp4CachePolicy;
    using kernel::block_scaled::Nvfp4Pipeline;
    p.nvfp4.pipeline = Nvfp4Pipeline::K256Deep;
    p.nvfp4.cache    = Nvfp4CachePolicy::BypassWeightL1;
    if (t > 16 && t <= 64) {
        p.gate_columns = p.down_columns = 16;
        p.nvfp4.cache                   = Nvfp4CachePolicy::BypassL1;
    } else if (t > 64 && t <= 128) {
        p.gate_columns          = 16;
        p.down_columns          = 8;
        p.nvfp4.pipeline        = Nvfp4Pipeline::K512;
        p.nvfp4.down_row_window = 1;
        p.queued_ctas_per_sm    = 12;
    } else if (t > 128 && t <= 256) {
        p.gate_columns = p.down_columns = 64;
        p.nvfp4.cache                   = Nvfp4CachePolicy::RetainInputs;
        p.nvfp4.down_row_window         = 4;
        p.queued_ctas_per_sm            = 12;
    } else if (t > 256 && t <= 768) {
        p.gate_columns          = 32;
        p.down_columns          = 32;
        p.nvfp4.pipeline        = Nvfp4Pipeline::K512;
        p.nvfp4.cache           = Nvfp4CachePolicy::RetainInputs;
        p.nvfp4.down_row_window = 4;
        p.queued_ctas_per_sm    = 4;
    } else if (t > 768 && t <= 1024) {
        p.gate_columns = p.down_columns = 32;
        p.nvfp4.pipeline                = Nvfp4Pipeline::K512;
        p.nvfp4.cache                   = Nvfp4CachePolicy::RetainInputs;
        p.nvfp4.down_row_window         = 4;
        p.queued_ctas_per_sm            = 8;
    } else if (t > 1024 && t <= 3072) {
        p.gate_columns          = 128;
        p.down_columns          = 64;
        p.nvfp4.pipeline        = Nvfp4Pipeline::K256;
        p.nvfp4.cache           = Nvfp4CachePolicy::RetainInputsBypassL1;
        p.nvfp4.down_row_window = 4;
        p.queued_ctas_per_sm    = t <= 2048 ? 2 : 12;
    } else if (t > 3072 && t <= 4096) {
        p.gate_columns = p.down_columns = 128;
        p.nvfp4.pipeline                = Nvfp4Pipeline::K128;
        p.nvfp4.cache                   = Nvfp4CachePolicy::RetainInputs;
        p.nvfp4.down_row_window         = 4;
        p.broad                         = {.gate_columns       = 128,
                                           .down_columns       = 128,
                                           .max_load           = 512,
                                           .tuning             = {Nvfp4Pipeline::K128, Nvfp4CachePolicy::Cached, 4},
                                           .queued_ctas_per_sm = 6};
        p.broad_short_limit             = 64;
        p.broad_long_limit              = 128;
    } else if (t > 4096) {
        p.gate_columns = p.down_columns = 128;
        p.nvfp4.pipeline                = Nvfp4Pipeline::K256;
        p.nvfp4.cache                   = Nvfp4CachePolicy::BypassInputL1;
        p.nvfp4.down_row_window         = 2;
    }
    if (t > 16) {
        p.dense.gate_columns           = t <= 64 ? 32 : t <= 128 ? 64 : 128;
        p.dense.tuning.pipeline        = t <= 32   ? Nvfp4Pipeline::K256Deep
                                         : t <= 64 ? Nvfp4Pipeline::K256
                                                   : Nvfp4Pipeline::K128;
        p.dense.tuning.down_row_window = t > 64 && t <= 512 ? 4 : 0;
        p.dense.queued_ctas_per_sm     = t > 32 && t <= 512 ? 12 : 6;
        p.small_expert_limit           = t <= 256    ? 16
                                         : t <= 1024 ? 32
                                         : t <= 3072 ? 64
                                         : t <= 4096 ? 128
                                                     : 256;
        if (t > 768 && t <= 1024) {
            p.medium.gate_columns = p.medium.down_columns = 64;
            p.medium.max_load                             = 64;
            p.medium.tuning.pipeline                      = Nvfp4Pipeline::K256;
            p.medium.tuning.cache                         = Nvfp4CachePolicy::RetainInputsBypassL1;
            p.medium.tuning.down_row_window               = 4;
            p.medium.queued_ctas_per_sm                   = 8;
        } else if (t > 1536 && t <= 3072) {
            p.medium.gate_columns = p.medium.down_columns = 128;
            p.medium.max_load                             = 128;
            p.medium.tuning.pipeline                      = Nvfp4Pipeline::K128;
            p.medium.tuning.cache                         = Nvfp4CachePolicy::RetainInputs;
            p.medium.tuning.down_row_window               = 4;
            p.medium.queued_ctas_per_sm                   = t <= 2048 ? 2 : 6;
        } else if (t > 4096) {
            p.medium.gate_columns = p.medium.down_columns = 128;
            p.medium.max_load                             = 512;
            p.medium.tuning.pipeline                      = Nvfp4Pipeline::K256;
            p.medium.tuning.cache                         = Nvfp4CachePolicy::BypassInputL1;
            p.medium.tuning.down_row_window               = 4;
        }
        p.dense.down_columns = p.dense.gate_columns;
        p.group_columns      = std::max(p.job_columns(), p.dense.gate_columns);
    }
    return plan;
}
} // namespace

const ShapePolicy& nvfp4_h4096_i2048_e288_k8() {
    static constexpr int boundaries[]{2,   17,   33,   65,   129,  257,  512,
                                      769, 1024, 1025, 1537, 2049, 3073, 4097};
    static const ShapePolicy policy{.select                = select_plan,
                                    .allocation_boundaries = boundaries,
                                    .all_slice_tokens      = 8192,
                                    .masked_slice_tokens   = 8192,
                                    .scaled_shared         = false,
                                    .accumulate            = false};
    return policy;
}
} // namespace ninfer::ops::detail::moe_routed
