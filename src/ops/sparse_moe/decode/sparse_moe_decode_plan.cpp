#include "ops/sparse_moe/decode/sparse_moe_decode.h"

#include "core/layout.h"

#include <stdexcept>

namespace ninfer::ops::detail {

std::size_t sparse_moe_decode_workspace_bytes() {
    WorkspaceLayoutBuilder layout;
    (void)allocate_sparse_moe_decode_workspace(layout);
    return layout.peak_bytes(256);
}

SparseMoeDecodePlan resolve_sparse_moe_decode_plan(QType routed_gate_up, QType routed_down) {
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
    return plan;
}

} // namespace ninfer::ops::detail
