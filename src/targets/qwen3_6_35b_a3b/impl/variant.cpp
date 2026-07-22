#include "targets/qwen3_6_35b_a3b/impl/variant.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"

#include <algorithm>
#include <stdexcept>

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_35b_a3b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_35b_a3b_runtime
#include "targets/qwen3_6/impl/runtime/instantiate.h"

namespace ninfer::targets::qwen3_6_35b_a3b::detail {
namespace {

std::vector<GraphFrontierRange>
graph_ranges_through(std::uint32_t max_frontier, const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphFrontierRange> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

void run_sparse_moe(const Tensor& hidden, const ops::SparseMoeWeights& weights, Tensor& residual,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope               = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(ops::sparse_moe_workspace_bytes(hidden.ne[1]));
    WorkspaceArena leaf_workspace(storage);
    ops::sparse_moe(hidden, weights, ops::SparseMoeEpilogue::AddResidual, residual, leaf_workspace,
                    stream);
}

} // namespace

std::vector<GraphFrontierRange> Variant::ordinary_graph_ranges(std::uint32_t capacity) {
    return graph_ranges_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
}

std::vector<GraphFrontierRange> Variant::mtp_graph_ranges(std::uint32_t capacity,
                                                          std::uint32_t draft_window) {
    if (draft_window == 0 || 2ULL * draft_window > capacity) { return {}; }
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, 2 * draft_window);
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_ranges_through(capacity - 2 * draft_window, ends);
}

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    ops::attn_input_proj(hidden, weights.query_key_gate_value, query, gate, key, value, workspace,
                         stream);
}

void Variant::mtp_attention_projection(const Tensor& hidden,
                                       const MtpAttentionProjectionWeights& weights, Tensor& query,
                                       Tensor& gate, Tensor& key, Tensor& value,
                                       WorkspaceArena& workspace, cudaStream_t stream) {
    ops::attn_input_proj(hidden, weights.query_key_gate_value, query, gate, key, value, workspace,
                         stream);
}

void Variant::mtp_kv_projection(const Tensor& hidden, const MtpAttentionProjectionWeights& weights,
                                Tensor& key, Tensor& value, WorkspaceArena& workspace,
                                cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor query   = workspace.alloc(DType::BF16, {TextConfig::query_size, cols});
    Tensor gate    = workspace.alloc(DType::BF16, {TextConfig::query_size, cols});
    ops::attn_input_proj(hidden, weights.query_key_gate_value, query, gate, key, value, workspace,
                         stream);
}

void Variant::mtp_q_gate_projection(const Tensor& hidden,
                                    const MtpAttentionProjectionWeights& weights, Tensor& query,
                                    Tensor& gate, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor key     = workspace.alloc(DType::BF16, {TextConfig::kv_size, cols});
    Tensor value   = workspace.alloc(DType::BF16, {TextConfig::kv_size, cols});
    ops::attn_input_proj(hidden, weights.query_key_gate_value, query, gate, key, value, workspace,
                         stream);
}

void Variant::gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                   Tensor& qkv, Tensor& output_gate, WorkspaceArena& workspace,
                                   cudaStream_t stream) {
    Tensor output_gate_flat =
        output_gate.view({TextConfig::value_dim, static_cast<int>(hidden.ne[1])});
    ops::gdn_input_proj(hidden, weights.query_key_value_z, qkv, output_gate_flat, workspace,
                        stream);
}

void Variant::gdn_input_projection_snapshot(const Tensor& hidden,
                                            const GdnProjectionWeights& weights,
                                            const Tensor& conv_weight, Tensor& conv_states,
                                            const Tensor& initial_slot, Tensor& query, Tensor& key,
                                            Tensor& value, Tensor& output_gate,
                                            WorkspaceArena& workspace, cudaStream_t stream) {
    Tensor output_gate_flat =
        output_gate.view({TextConfig::value_dim, static_cast<int>(hidden.ne[1])});
    ops::gdn_input_proj_conv_snapshot(hidden, weights.query_key_value_z, conv_weight, conv_states,
                                      initial_slot, query, key, value, output_gate_flat, workspace,
                                      stream);
}

void Variant::gdn_control_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                     Tensor& g, Tensor& beta, WorkspaceArena& workspace,
                                     cudaStream_t stream) {
    ops::gdn_gating_proj(hidden, weights.a_b_projection, weights.a_log, weights.dt_bias, workspace,
                         g, beta, stream);
}

void Variant::gdn_output_gate_projection(const Tensor&, const GdnProjectionWeights&, Tensor&,
                                         WorkspaceArena&, cudaStream_t) {}

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         WorkspaceArena& workspace, cudaStream_t stream) {
    run_sparse_moe(hidden, weights.op, residual, workspace, stream);
}

void Variant::mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                             Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream) {
    run_sparse_moe(hidden, weights.op, residual, workspace, stream);
}

std::size_t Variant::mtp_attention_workspace_bytes(std::int32_t) { return 0; }

std::size_t Variant::mtp_kv_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::query_size, tokens});
    (void)layout.alloc(DType::BF16, {TextConfig::query_size, tokens});
    return layout.peak_bytes();
}

std::size_t Variant::mtp_q_gate_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::kv_size, tokens});
    (void)layout.alloc(DType::BF16, {TextConfig::kv_size, tokens});
    return layout.peak_bytes();
}

std::size_t Variant::gdn_input_projection_workspace_bytes(std::int32_t tokens) {
    return ops::gdn_input_proj_workspace_bytes(TextConfig::convolution_dim, TextConfig::value_dim,
                                               tokens);
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_bytes(std::int32_t tokens) {
    return ops::gdn_input_proj_conv_snapshot_workspace_bytes(
        TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, tokens);
}

std::size_t Variant::gdn_control_projection_workspace_bytes(std::int32_t tokens) {
    return ops::gdn_gating_proj_workspace_bytes(tokens);
}

std::size_t Variant::gdn_output_gate_projection_workspace_bytes(std::int32_t) { return 0; }

std::size_t Variant::post_mixer_workspace_bytes(std::int32_t tokens) {
    return ops::sparse_moe_workspace_bytes(tokens);
}

std::size_t Variant::mtp_post_mixer_workspace_bytes(std::int32_t tokens) {
    return ops::sparse_moe_workspace_bytes(tokens);
}

} // namespace ninfer::targets::qwen3_6_35b_a3b::detail
