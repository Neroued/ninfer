#include "ops/moe/routed_swiglu/cuda/dispatch.h"

namespace ninfer::ops::detail::moe_routed {
namespace {
Plan select_plan(int tokens, int) {
    Plan plan;
    plan.implementation = kernel::ExpertImplementation::BlockScaled;
    const int columns   = tokens <= 16 ? 8 : tokens <= 128 ? 32 : 64;
    plan.block_scaled   = {.format                = QType::NVFP4,
                           .gate_columns          = tokens >= 3584 ? 128 : columns,
                           .down_columns          = columns,
                           .vector                = tokens == 1,
                           .atomic_output         = tokens >= 128,
                           .single_block_grouping = tokens <= 64,
                           .shared_algorithm      = tokens < 8
                                                        ? kernel::block_scaled::SharedAlgorithm::Vector
                                                        : kernel::block_scaled::SharedAlgorithm::Mma,
                           .queued_ctas_per_sm    = 6};
    return plan;
}
} // namespace

const ShapePolicy& nvfp4_h2560_i640_e512_k10() {
    static constexpr int boundaries[]{2, 17, 65, 128, 129, 3584};
    static const ShapePolicy policy{.select = select_plan, .allocation_boundaries = boundaries};
    return policy;
}
} // namespace ninfer::ops::detail::moe_routed
