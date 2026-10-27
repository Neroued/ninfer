#include "ops/moe/routed_swiglu/cuda/dispatch.h"

namespace ninfer::ops::detail::moe_routed {
namespace {
Plan select_plan(const SwiGLUProfile&, int t, int shared_intermediate) {
    Plan plan;
    plan.implementation = Implementation::BlockScaled;
    auto& p             = plan.block_scaled;
    p                   = {.format                  = QType::NVFP4,
                           .intermediate            = 320,
                           .gate_columns            = 64,
                           .down_columns            = 64,
                           .vector                  = t <= 3,
                           .atomic_output           = true,
                           .single_block_grouping   = t <= 64,
                           .column_major_down       = t >= 1024,
                           .preserve_balanced_down  = true,
                           .shared_algorithm        = t < 8 ? kernel::block_scaled::SharedAlgorithm::Vector
                                                            : kernel::block_scaled::SharedAlgorithm::Mma,
                           .shared_split_activation = shared_intermediate != 320 || t < 128,
                           .network_topk            = false,
                           .shared_columns          = shared_intermediate == 320 && t >= 512 ? 64 : 32,
                           .queued_ctas_per_sm      = 6};
    using kernel::block_scaled::Nvfp4Pipeline;
    using kernel::block_scaled::Nvfp4CachePolicy;
    p.nvfp4 = {Nvfp4Pipeline::K512, Nvfp4CachePolicy::RetainInputs, 4};
    if (t > 4) {
        p.small_expert_limit = t <= 64 ? 4 : t <= 512 ? 16 : 128;
        if (t <= 64)
            p.medium = {.gate_columns = 16,
                        .down_columns = 16,
                        .max_load     = 64,
                        .tuning = {Nvfp4Pipeline::K256Deep, Nvfp4CachePolicy::BypassWeightL1, 0},
                        .queued_ctas_per_sm = 6};
        p.dense         = {.gate_columns       = t <= 512 ? 64 : 128,
                           .down_columns       = t <= 512 ? 64 : 128,
                           .tuning             = {Nvfp4Pipeline::K128, Nvfp4CachePolicy::Cached, 0},
                           .queued_ctas_per_sm = t <= 512 ? 6 : 2};
        p.group_columns = t <= 512 ? 64 : 128;
    }
    return plan;
}
} // namespace

const ShapePolicy& nvfp4_h2560_i320_e512_k10() {
    static constexpr int boundaries[]{4, 5, 8, 65, 128, 512, 513, 1024};
    static const ShapePolicy policy{.select = select_plan, .allocation_boundaries = boundaries};
    return policy;
}
} // namespace ninfer::ops::detail::moe_routed
