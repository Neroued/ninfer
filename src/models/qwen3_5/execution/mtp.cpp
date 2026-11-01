#include "models/qwen3_5/execution/mtp.h"

#include "core/layout.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/mtp_pack.h"

#include <algorithm>

namespace ninfer::models::qwen3_5::execution {

std::size_t mtp_projection_workspace_bytes(const MtpProjectionParameters& parameters,
                                           std::int32_t first, std::int32_t last) {
    const auto& p = parameters.packed;
    const auto& w = p.weight;
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {w.n, last});
    (void)layout.alloc_bytes(
        ops::linear_workspace_capacity_bytes(w.qtype, w.n, w.k, p.policy, first, last));
    return layout.peak_bytes(1);
}

std::size_t mtp_kv_workspace_bytes(const MtpProjectionParameters& parameters, std::int32_t first,
                                   std::int32_t last) {
    return ops::linear_pair_workspace_capacity_bytes(parameters.rows[1].weight,
                                                     parameters.rows[3].weight, first, last);
}

std::size_t mtp_query_gate_workspace_bytes(const MtpProjectionParameters& parameters,
                                           std::int32_t first, std::int32_t last) {
    const auto bytes = [&](std::size_t index) {
        const auto& p = parameters.rows[index];
        const auto& w = p.weight;
        return ops::linear_workspace_capacity_bytes(w.qtype, w.n, w.k, p.policy, first, last);
    };
    return std::max(bytes(0), bytes(2));
}

void mtp_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                    const AttentionConfig& config, Tensor& query, Tensor& gate, Tensor& key,
                    Tensor& value, WorkspaceArena& workspace, cudaStream_t stream) {
    const auto& p      = parameters.packed;
    auto scope         = workspace.scope();
    const auto columns = hidden.ne[1];
    Tensor packed      = workspace.alloc(DType::BF16, {p.weight.n, columns});
    ops::linear(hidden, p.weight, packed, p.policy, workspace, stream);
    Tensor q =
        query.view({dimension(config.head_dim), dimension(config.num_attention_heads), columns});
    Tensor k =
        key.view({dimension(config.head_dim), dimension(config.num_key_value_heads), columns});
    Tensor g =
        gate.view({dimension(config.head_dim), dimension(config.num_attention_heads), columns});
    Tensor v =
        value.view({dimension(config.head_dim), dimension(config.num_key_value_heads), columns});
    ops::mtp_split_attn_in(packed, q, k, g, v, stream);
}

void mtp_kv_projection(const Tensor& hidden, const MtpProjectionParameters& parameters, Tensor& key,
                       Tensor& value, cudaStream_t stream) {
    ops::linear_pair(hidden, parameters.rows[1].weight, parameters.rows[3].weight, key, value,
                     stream);
}

void mtp_query_gate_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                               Tensor& query, Tensor& gate, WorkspaceArena& workspace,
                               cudaStream_t stream) {
    const auto& q = parameters.rows[0];
    const auto& g = parameters.rows[2];
    {
        auto scope = workspace.scope();
        ops::linear(hidden, q.weight, query, q.policy, workspace, stream);
    }
    ops::linear(hidden, g.weight, gate, g.policy, workspace, stream);
}

} // namespace ninfer::models::qwen3_5::execution
