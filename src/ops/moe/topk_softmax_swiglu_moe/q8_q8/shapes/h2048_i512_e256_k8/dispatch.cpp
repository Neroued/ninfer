#include "ops/moe/topk_softmax_swiglu_moe/dispatch.h"

#include <array>

namespace ninfer::ops::detail::moe_complete {
namespace {
constexpr int kGroupedFirst = 20;
constexpr int kWideFirst    = 384;
constexpr std::array kAllocationBoundaries{2, kGroupedFirst, kWideFirst};

Plan select_plan(int tokens, int, ExpertSubsetKind subset) {
    Plan plan;
    plan.integer.grouped = tokens >= kGroupedFirst;
    plan.router          = {.algorithm         = tokens == 1            ? RouterAlgorithm::Vector
                                                 : plan.integer.grouped ? RouterAlgorithm::Mma
                                                                        : RouterAlgorithm::TokenTile,
                            .programmatic      = !plan.integer.grouped,
                            .count_assignments = plan.integer.grouped && subset == ExpertSubsetKind::All};
    plan.integer.token   = {.paths_per_block    = tokens == 1   ? 9
                                                  : tokens <= 5 ? 1
                                                                : 9,
                            .output_rows        = tokens == 1   ? 1
                                                  : tokens <= 8 ? 1
                                                                : 4,
                            .single_token       = tokens == 1,
                            .programmatic_input = true};
    plan.integer.group   = {.columns            = tokens >= kWideFirst ? 64 : 32,
                            .adaptive           = false,
                            .spread_numerator   = 7,
                            .spread_denominator = 4,
                            .queued_ctas_per_sm = 32,
                            .shared_columns     = 64,
                            .direct_numerator   = 7,
                            .direct_denominator = 16};
    plan.integer.shared  = {.token   = {.paths_per_block = 1,
                                        .output_rows     = tokens == 1 ? 1 : 4,
                                        .single_token    = tokens == 1},
                            .columns = tokens >= 16 ? 64 : 0};
    return plan;
}
} // namespace

const ShapePolicy& q8_q8_h2048_i512_e256_k8() {
    static const ShapePolicy policy{.select                = select_plan,
                                    .allocation_boundaries = kAllocationBoundaries};
    return policy;
}
} // namespace ninfer::ops::detail::moe_complete
