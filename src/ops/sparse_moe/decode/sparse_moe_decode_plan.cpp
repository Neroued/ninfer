#include "ops/sparse_moe/decode/sparse_moe_decode.h"

#include "core/layout.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
// The D4 epilogue warms at most this many bytes of the next consumer's weights: far below the
// 96 MiB L2, so the warmed block cannot evict what it was meant to help.
constexpr std::size_t kNextWeightPrefetchLimit = std::size_t{8} << 20;
} // namespace

std::size_t sparse_moe_decode_workspace_bytes() {
    WorkspaceLayoutBuilder layout;
    (void)allocate_sparse_moe_decode_workspace(layout);
    return layout.peak_bytes(1);
}

SparseMoeDecodePlan resolve_sparse_moe_decode_plan(QType routed_gate_up, QType routed_down,
                                                   const SparseMoeHints& hints) {
    const bool main_profile =
        routed_gate_up == QType::Q4G64_F16S &&
        (routed_down == QType::Q5G64_F16S || routed_down == QType::Q6G64_F16S);
    const bool mtp_profile =
        routed_gate_up == QType::W8G32_F16S && routed_down == QType::W8G32_F16S;
    if (!main_profile && !mtp_profile) {
        throw std::invalid_argument("sparse_moe: unsupported routed codec profile");
    }

    SparseMoeDecodePlan plan;
    plan.workspace_bytes = sparse_moe_decode_workspace_bytes();
    if (hints.next_weight_prefetch != nullptr) {
        plan.next_weight_prefetch = hints.next_weight_prefetch;
        plan.next_weight_prefetch_bytes =
            hints.next_weight_prefetch_bytes < kNextWeightPrefetchLimit
                ? hints.next_weight_prefetch_bytes
                : kNextWeightPrefetchLimit;
    }
    return plan;
}

} // namespace ninfer::ops::detail
