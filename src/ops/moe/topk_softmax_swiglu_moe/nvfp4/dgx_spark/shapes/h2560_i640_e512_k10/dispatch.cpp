#include "ops/moe/topk_softmax_swiglu_moe/dispatch.h"

namespace ninfer::ops::detail::moe_complete {
namespace {
Plan select_plan(int tokens, int shared_intermediate, ExpertSubsetKind) {
    Plan plan;
    plan.implementation = kernel::ExpertImplementation::BlockScaled;
    const int columns   = tokens <= 16 ? 8 : tokens <= 128 ? 32 : tokens < 512 ? 64 : 128;
    plan.block_scaled   = {.format                  = QType::NVFP4,
                           .gate_columns            = columns,
                           .down_columns            = columns,
                           .vector                  = tokens == 1,
                           .atomic_output           = tokens >= 128,
                           .single_block_grouping   = tokens <= 64,
                           .column_major_down       = tokens >= 1024,
                           .preserve_balanced_down  = true,
                           .shared_algorithm        = tokens < 8
                                                          ? kernel::block_scaled::SharedAlgorithm::Vector
                                                          : kernel::block_scaled::SharedAlgorithm::Mma,
                           .shared_split_activation = shared_intermediate != 320 || tokens < 128,
                           .network_topk            = true,
                           .shared_columns          = tokens >= 512 ? 64 : 32,
                           .queued_ctas_per_sm      = 6};
    return plan;
}
} // namespace

const ShapePolicy& nvfp4_h2560_i640_e512_k10() {
    static constexpr int boundaries[]{2, 17, 65, 128, 129, 512};
    static const ShapePolicy policy{.select = select_plan, .allocation_boundaries = boundaries};
    return policy;
}
} // namespace ninfer::ops::detail::moe_complete
