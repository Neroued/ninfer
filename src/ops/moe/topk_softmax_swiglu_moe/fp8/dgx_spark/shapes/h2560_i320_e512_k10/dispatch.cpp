#include "ops/moe/topk_softmax_swiglu_moe/dispatch.h"

namespace ninfer::ops::detail::moe_complete {
namespace {
Plan select_plan(int t, int shared_intermediate, ExpertSubsetKind) {
    Plan plan;
    plan.implementation = kernel::ExpertImplementation::BlockScaled;
    auto& p             = plan.block_scaled;
    p                   = {.format                  = QType::FP8_E4M3FN_BLOCK64X64_BF16,
                           .intermediate            = 320,
                           .gate_columns            = 16,
                           .down_columns            = 16,
                           .vector                  = t <= 2,
                           .atomic_output           = true,
                           .single_block_grouping   = t <= 64,
                           .column_major_down       = t >= 1024,
                           .preserve_balanced_down  = true,
                           .shared_algorithm        = t < 8 ? kernel::block_scaled::SharedAlgorithm::Vector
                                                            : kernel::block_scaled::SharedAlgorithm::Mma,
                           .shared_split_activation = shared_intermediate != 320 || t < 128,
                           .network_topk            = true,
                           .shared_columns          = shared_intermediate == 320 && t >= 512 ? 64 : 32,
                           .queued_ctas_per_sm      = 12};
    if (t >= 17)
        p.fp8 = {.maximum_small_load       = 16,
                 .large_gate_columns       = t < 128 || (t > 128 && t <= 256) ? 64 : 128,
                 .large_down_columns       = 64,
                 .large_queued_ctas_per_sm = 6,
                 .maximum_large_load       = t >= 2048 ? 128 : 0,
                 .dense_queued_ctas_per_sm = 3};
    if (t >= 32 && t <= 512) {
        p.gate_columns = p.down_columns = 64;
        p.fp8                           = {};
        p.queued_ctas_per_sm            = 6;
    }
    if (t >= 256) {
        p.pool_expert_limit       = t < 2048 ? 10 : 20;
        p.pool_columns            = 64;
        p.pool_row_window         = 4;
        p.pool_queued_ctas_per_sm = 3;
    }
    return plan;
}
} // namespace

const ShapePolicy& fp8_h2560_i320_e512_k10() {
    static constexpr int boundaries[]{3, 8, 17, 32, 65, 128, 129, 256, 257, 512, 513, 1024, 2048};
    static const ShapePolicy policy{.select = select_plan, .allocation_boundaries = boundaries};
    return policy;
}
} // namespace ninfer::ops::detail::moe_complete
