#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"

#include "core/layout.h"

#include <stdexcept>

namespace ninfer::ops::detail {

bool sparse_moe_uses_small_t(std::int32_t tokens) noexcept {
    return tokens >= kSparseMoeSmallTMin && tokens <= kSparseMoeSmallTMax;
}

std::size_t sparse_moe_small_t_workspace_bytes(std::int32_t tokens) {
    if (!sparse_moe_uses_small_t(tokens)) {
        throw std::invalid_argument("sparse_moe small-T: tokens must be in [2,46]");
    }
    WorkspaceLayoutBuilder layout;
    (void)allocate_sparse_moe_small_t_workspace(layout, tokens);
    return layout.peak_bytes(1);
}

SparseMoeSmallTPlan resolve_sparse_moe_small_t_plan(std::int32_t tokens, QType routed_gate_up,
                                                    QType routed_down) {
    if (!sparse_moe_uses_small_t(tokens)) {
        throw std::invalid_argument("sparse_moe small-T: unsupported token count");
    }
    const bool main_profile =
        routed_gate_up == QType::Q4G64_F16S &&
        (routed_down == QType::Q5G64_F16S || routed_down == QType::Q6G64_F16S);
    const bool mtp_profile =
        routed_gate_up == QType::W8G32_F16S && routed_down == QType::W8G32_F16S;
    if (!main_profile && !mtp_profile) {
        throw std::invalid_argument("sparse_moe small-T: unsupported routed codec profile");
    }

    SparseMoeSmallTPlan plan{tokens, sparse_moe_small_t_workspace_bytes(tokens)};
    if (mtp_profile) {
        plan.d3_schedule =
            tokens <= 5 ? SparseMoeSmallTD3Schedule::Paths1 : SparseMoeSmallTD3Schedule::Paths9;
        plan.d4_schedule =
            tokens <= 8 ? SparseMoeSmallTD4Schedule::Rows1 : SparseMoeSmallTD4Schedule::Rows4;
        return plan;
    }

    plan.d3_schedule = SparseMoeSmallTD3Schedule::Paths3;
    // Rows is only a grid split: (kHidden / Rows, tokens). Rows2 runs twice the blocks, so it has
    // twice the parallelism to cover DRAM latency while the grid is small, and it re-reads the
    // same per-expert activations in every block - measured at +63 % L1 traffic with L2 and DRAM
    // traffic unchanged. Below the crossover the parallelism wins; above it only the cost is left.
    // Each codec carries its own measured crossover, taken on the routing pattern where Rows2 is
    // weakest (all tokens sharing one set of eight experts): 17 tokens for Q5G64_F16S, 11 for
    // Q6G64_F16S. Both hold at 1975 MHz and at 2810 MHz; the crossover moves right with SM clock,
    // because the cost is SM-side, so the lower-clocked reading is the one that binds.
    if (routed_down == QType::Q5G64_F16S) {
        plan.d4_schedule = tokens <= 2    ? SparseMoeSmallTD4Schedule::Rows1
                           : tokens <= 17 ? SparseMoeSmallTD4Schedule::Rows2
                                          : SparseMoeSmallTD4Schedule::Rows4;
    } else {
        plan.d4_schedule = tokens <= 2    ? SparseMoeSmallTD4Schedule::Rows1
                           : tokens <= 11 ? SparseMoeSmallTD4Schedule::Rows2
                                          : SparseMoeSmallTD4Schedule::Rows4;
    }
    return plan;
}

} // namespace ninfer::ops::detail
