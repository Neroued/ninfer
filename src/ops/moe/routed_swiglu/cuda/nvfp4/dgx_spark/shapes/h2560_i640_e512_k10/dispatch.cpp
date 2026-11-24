#include "ops/moe/routed_swiglu/cuda/dispatch.h"

#include <algorithm>

namespace ninfer::ops::detail::moe_routed {
namespace {
using SharedAlgorithm = kernel::block_scaled::SharedAlgorithm;

Plan select_plan(const SwiGLUProfile&, int tokens, int shared_intermediate) {
    Plan plan;
    plan.implementation = Implementation::BlockScaled;
    plan.block_scaled   = {.format                 = QType::NVFP4,
                           .gate_columns           = 8,
                           .down_columns           = 8,
                           .vector                 = tokens == 1,
                           .atomic_output          = tokens >= 128,
                           .single_block_grouping  = tokens <= 64,
                           .column_major_down      = tokens >= 1024,
                           .preserve_balanced_down = true,
                           .shared_algorithm =
                             tokens < 8 ? SharedAlgorithm::Vector : SharedAlgorithm::Mma,
                           .shared_split_activation = shared_intermediate != 320 || tokens < 128,
                           .shared_columns = shared_intermediate == 320 && tokens >= 512 ? 64 : 32,
                           .queued_ctas_per_sm = 6};
    auto& p             = plan.block_scaled;
    using kernel::block_scaled::Nvfp4CachePolicy;
    using kernel::block_scaled::Nvfp4Pipeline;
    p.nvfp4.pipeline = Nvfp4Pipeline::K256Deep;
    p.nvfp4.cache    = Nvfp4CachePolicy::BypassWeightL1;
    if (tokens > 16 && tokens <= 64) {
        p.gate_columns = 32;
        p.down_columns = 16;
    } else if (tokens > 64 && tokens < 512) {
        p.gate_columns       = 16;
        p.down_columns       = 32;
        p.nvfp4.cache        = Nvfp4CachePolicy::StreamingAll;
        p.queued_ctas_per_sm = 12;
    } else if (tokens >= 512 && tokens <= 1024) {
        p.gate_columns   = 128;
        p.down_columns   = 32;
        p.nvfp4.pipeline = Nvfp4Pipeline::K512;
        p.nvfp4.cache =
            tokens <= 768 ? Nvfp4CachePolicy::RetainInputs : Nvfp4CachePolicy::StreamingAll;
        p.nvfp4.down_row_window = tokens <= 768 ? 0 : 4;
        p.queued_ctas_per_sm    = tokens <= 768 ? 12 : 6;
    } else if (tokens > 1024 && tokens <= 2048) {
        p.gate_columns          = 128;
        p.down_columns          = 64;
        p.nvfp4.pipeline        = Nvfp4Pipeline::K128;
        p.nvfp4.cache           = Nvfp4CachePolicy::StreamingWeights;
        p.nvfp4.down_row_window = 4;
        p.queued_ctas_per_sm    = 4;
    } else if (tokens > 2048) {
        p.gate_columns = p.down_columns = 128;
        p.nvfp4.pipeline                = Nvfp4Pipeline::K128;
        p.nvfp4.cache                   = Nvfp4CachePolicy::StreamingAll;
        p.nvfp4.down_row_window         = 4;
        // Match the measured two-CTA residency of this tile.
        p.queued_ctas_per_sm = tokens <= 4096 ? 2 : 6;
    }
    if (tokens > 16) {
        p.dense.gate_columns       = tokens <= 64 ? 32 : tokens < 512 ? 64 : 128;
        p.dense.tuning.pipeline    = tokens <= 32 ? Nvfp4Pipeline::K256Deep : Nvfp4Pipeline::K128;
        p.dense.queued_ctas_per_sm = tokens > 128 && tokens <= 512 ? 12 : 6;
        p.small_expert_limit = tokens < 512 ? 16 : tokens <= 1024 ? 32 : tokens <= 2048 ? 64 : 128;
        if (tokens >= 512) {
            p.medium.gate_columns           = 128;
            p.medium.down_columns           = tokens <= 1024 ? 64 : tokens <= 2048 ? 128 : 32;
            p.medium.max_load               = tokens <= 1024 ? 128 : 512;
            p.medium.tuning.cache           = tokens <= 1024   ? Nvfp4CachePolicy::StreamingWeights
                                              : tokens <= 2048 ? Nvfp4CachePolicy::StreamingAll
                                                               : Nvfp4CachePolicy::BypassL1;
            p.medium.tuning.down_row_window = tokens <= 2048 ? 4 : 2;
        }
        p.dense.down_columns = p.dense.gate_columns;
        p.group_columns = tokens > 4096 ? 256 : std::max(p.job_columns(), p.dense.gate_columns);
    }
    return plan;
}
} // namespace

const ShapePolicy& nvfp4_h2560_i640_e512_k10() {
    static constexpr int boundaries[]{2, 17, 33, 65, 128, 129, 512, 769, 1025, 2049, 4097};
    static const ShapePolicy policy{
        .select = select_plan, .allocation_boundaries = boundaries, .masked_slice_tokens = 8192};
    return policy;
}
} // namespace ninfer::ops::detail::moe_routed
