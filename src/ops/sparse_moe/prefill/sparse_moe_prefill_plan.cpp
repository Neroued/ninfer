#include "core/weight.h"
#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"

#include "core/layout.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

std::int32_t prefill_min_tokens(QType routed_gate_up, QType routed_down) noexcept {
    if (routed_gate_up == QType::Q4_G64_FP16) {
        if (routed_down == QType::Q5_G64_FP16) { return kSparseMoePrefillQ4Q5Min; }
        if (routed_down == QType::Q6_G64_FP16) { return kSparseMoePrefillQ4Q6Min; }
    }
    if (routed_gate_up == QType::Q8_G32_FP16 && routed_down == QType::Q8_G32_FP16) {
        return kSparseMoePrefillQ8Q8Min;
    }
    return 0;
}

} // namespace

bool sparse_moe_uses_prefill(std::int32_t tokens, QType routed_gate_up,
                             QType routed_down) noexcept {
    const std::int32_t minimum = prefill_min_tokens(routed_gate_up, routed_down);
    return minimum != 0 && tokens >= minimum;
}

std::size_t sparse_moe_prefill_workspace_bytes(std::int32_t max_tokens) {
    if (max_tokens < kSparseMoePrefillWorkspaceMin) {
        throw std::invalid_argument("sparse_moe prefill: max_tokens must be at least 20");
    }
    const std::int32_t capacity_tokens = std::min(max_tokens, kSparseMoePrefillSliceMax);
    WorkspaceLayoutBuilder layout;
    (void)allocate_sparse_moe_prefill_workspace(layout, capacity_tokens);
    return layout.peak_bytes(1);
}

SparseMoePrefillPlan resolve_sparse_moe_prefill_plan(std::int32_t tokens, QType routed_gate_up,
                                                     QType routed_down) {
    const std::int32_t minimum = prefill_min_tokens(routed_gate_up, routed_down);
    if (minimum == 0) {
        throw std::invalid_argument("sparse_moe prefill: unsupported routed codec profile");
    }
    if (tokens < minimum) {
        throw std::invalid_argument("sparse_moe prefill: unsupported token count");
    }

    const std::int32_t slice_tokens = std::min(tokens, kSparseMoePrefillSliceMax);
    return {tokens, slice_tokens, sparse_moe_prefill_workspace_bytes(tokens)};
}

} // namespace ninfer::ops::detail
