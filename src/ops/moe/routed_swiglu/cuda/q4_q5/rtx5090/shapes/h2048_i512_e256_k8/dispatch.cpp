#include "ops/moe/routed_swiglu/cuda/dispatch.h"

#include <array>

namespace ninfer::ops::detail::moe_routed {
namespace {
constexpr int kGroupedFirst = 47;
constexpr std::array kAllocationBoundaries{kGroupedFirst, 768};

Plan select_plan(const SwiGLUProfile&, int tokens, int shared_intermediate) {
    const bool shared = shared_intermediate != 0;
    Plan plan;
    plan.integer.grouped = tokens >= kGroupedFirst;
    plan.integer.token   = {.paths_per_block    = tokens == 1 ? (shared ? 9 : 8) : (shared ? 3 : 4),
                            .output_rows        = tokens <= 2    ? 1
                                                  : tokens <= 17 ? 2
                                                                 : 4,
                            .single_token       = tokens == 1,
                            .programmatic_input = false};
    plan.integer.group   = {.columns            = tokens >= 768 ? 64 : 32,
                            .adaptive           = false,
                            .spread_numerator   = 7,
                            .spread_denominator = 4,
                            .queued_ctas_per_sm = 32,
                            .shared_columns     = 64,
                            .direct_numerator   = 7,
                            .direct_denominator = 16};
    plan.integer.group.adaptive = plan.integer.grouped && tokens <= 128;
    if (plan.integer.group.adaptive) {
        plan.integer.token = {.paths_per_block    = shared ? 3 : 4,
                              .output_rows        = 4,
                              .single_token       = false,
                              .programmatic_input = false};
    }
    plan.integer.shared = {.token   = {.paths_per_block    = 1,
                                       .output_rows        = tokens == 1 ? 1 : 4,
                                       .single_token       = tokens == 1,
                                       .programmatic_input = false},
                           .columns = tokens >= 16 ? 64 : 0};
    return plan;
}
} // namespace

const ShapePolicy& q4_q5_h2048_i512_e256_k8() {
    static const ShapePolicy policy{.select                = select_plan,
                                    .allocation_boundaries = kAllocationBoundaries};
    return policy;
}
} // namespace ninfer::ops::detail::moe_routed
