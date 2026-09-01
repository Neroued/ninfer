#include "targets/qwen3_6_35b_a3b/impl/variant.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/sparse_moe.h"

#include <algorithm>
#include <stdexcept>

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_35b_a3b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_35b_a3b_runtime
#include "targets/qwen3_6/impl/runtime/instantiate.h"

namespace ninfer::targets::qwen3_6_35b_a3b::detail {
namespace {

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
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

std::vector<GraphExecutionProfile> dflash_base_profiles(std::uint32_t capacity,
                                                        std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    const std::uint32_t block        = draft_window + 1;
    const std::uint32_t max_frontier = capacity - 1;
    std::vector<std::uint32_t> ends{
        96U, 127U, 511U, 1023U, 2047U, 4095U, 8191U, 16383U, 32767U, 65536U, 131072U, 196608U,
    };
    const auto add_target_boundary = [&](std::uint32_t visible_end) {
        if (visible_end >= block) { ends.push_back(visible_end - block); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_target_boundary(visible_end);
    }
    if (draft_window >= 6 && draft_window <= 15) {
        add_target_boundary(draft_window <= 11 ? 512U : 1024U);
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_profiles_through(max_frontier, ends);
}

bool dflash_target_uses_chunked_small_t(std::uint32_t draft_window, std::uint32_t batch_size,
                                        std::uint32_t max_visible_keys) {
    const std::uint32_t tokens = draft_window + 1;
    if (tokens <= 6) { return false; }
    if (batch_size > 1) { return true; }
    const std::uint32_t prompt_visible_limit = tokens <= 12 ? 512U : 1024U;
    return max_visible_keys > prompt_visible_limit;
}

void run_sparse_moe(const Tensor& hidden, const ops::SparseMoeWeights& weights, Tensor& residual,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope               = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(ops::sparse_moe_workspace_capacity_bytes(
        weights.routed_gate_up.qtype, weights.routed_down.qtype, hidden.ne[1], hidden.ne[1]));
    WorkspaceArena leaf_workspace(storage);
    ops::sparse_moe(hidden, weights, ops::SparseMoeEpilogue::AddResidual, residual, leaf_workspace,
                    stream);
}

void validate_token_interval(std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("invalid target leaf token interval");
    }
}

constexpr std::size_t kMinimumLeafWorkspaceBytes = 1;

std::size_t gdn_record_workspace_bytes(const Tensor& hidden) {
    return std::max(kMinimumLeafWorkspaceBytes,
                    ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                        TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim,
                        hidden.ne[2], hidden.ne[1], hidden.ne[1]));
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    return graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t capacity,
                                                               std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, 2 * draft_window);
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_profiles_through(capacity - 1, ends);
}

std::vector<GraphExecutionProfile> Variant::dflash_graph_profiles(std::uint32_t capacity,
                                                                  std::uint32_t draft_window,
                                                                  std::uint32_t batch_size) {
    std::vector<GraphExecutionProfile> profiles = dflash_base_profiles(capacity, draft_window);
    for (GraphExecutionProfile& profile : profiles) {
        const std::uint32_t target_max = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            capacity, static_cast<std::uint64_t>(profile.max) + draft_window + 1ULL));
        const bool split_swa           = profile.max > 96U;
        const bool chunked_target =
            dflash_target_uses_chunked_small_t(draft_window, batch_size, target_max);
        profile.topology_class = (chunked_target ? 2U : 0U) | (split_swa ? 1U : 0U);
    }
    return profiles;
}

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value, qwen3_6::TextPhase,
                                   WorkspaceArena&, cudaStream_t stream) {
    ops::attn_input_proj(hidden, weights.query_key_gate_value, query, gate, key, value, stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, qwen3_6::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear_add(attention, weight, residual, workspace, stream);
}

void Variant::mtp_attention_projection(const Tensor& hidden,
                                       const MtpAttentionProjectionWeights& weights, Tensor& query,
                                       Tensor& gate, Tensor& key, Tensor& value,
                                       WorkspaceArena& workspace, cudaStream_t stream) {
    ops::attn_input_proj(hidden, weights.query_key_gate_value, query, gate, key, value, stream);
}

void Variant::mtp_kv_projection(const Tensor& hidden, const MtpAttentionProjectionWeights& weights,
                                Tensor& key, Tensor& value, WorkspaceArena& workspace,
                                cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor query   = workspace.alloc(DType::BF16, {TextConfig::query_size, cols});
    Tensor gate    = workspace.alloc(DType::BF16, {TextConfig::query_size, cols});
    ops::attn_input_proj(hidden, weights.query_key_gate_value, query, gate, key, value, stream);
}

void Variant::mtp_q_gate_projection(const Tensor& hidden,
                                    const MtpAttentionProjectionWeights& weights, Tensor& query,
                                    Tensor& gate, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor key     = workspace.alloc(DType::BF16, {TextConfig::kv_size, cols});
    Tensor value   = workspace.alloc(DType::BF16, {TextConfig::kv_size, cols});
    ops::attn_input_proj(hidden, weights.query_key_gate_value, query, gate, key, value, stream);
}

void Variant::gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                   Tensor& qkv, Tensor& output_gate, qwen3_6::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    Tensor output_gate_flat =
        output_gate.view({TextConfig::value_dim, static_cast<int>(hidden.ne[1] * hidden.ne[2])});
    // The policy-bearing overload, not the convenience one: this leaf is handed a
    // workspace and the wide-T route needs it to stage a dequantized parent. The
    // capacity is already reported by
    // gdn_input_projection_workspace_capacity_bytes() below, which forwards to the
    // Op's own query.
    ops::gdn_input_proj(hidden, weights.query_key_value_z, qkv, output_gate_flat,
                        ops::LinearPolicy::A16Only, workspace, stream);
}

void Variant::gdn_input_projection_snapshot(
    const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot,
    const Tensor& snapshot_base_slot, Tensor& query, Tensor& key, Tensor& value,
    Tensor& output_gate, qwen3_6::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    Tensor output_gate_view = output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    ops::gdn_input_proj_conv_snapshot(hidden, weights.query_key_value_z, conv_weight, conv_states,
                                      valid_columns, initial_slot, snapshot_base_slot, query, key,
                                      value, output_gate_view, workspace, stream);
}

void Variant::gdn_input_projection_record(const Tensor& hidden, const GdnProjectionWeights& weights,
                                          const Tensor& conv_weight, const Tensor& conv_states,
                                          const Tensor& valid_columns, const Tensor& initial_slots,
                                          Tensor& conv_record, Tensor& query, Tensor& key,
                                          Tensor& value, Tensor& output_gate, qwen3_6::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    auto workspace_scope     = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(gdn_record_workspace_bytes(hidden));
    WorkspaceArena leaf_workspace(storage);
    Tensor output_gate_view = output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    ops::gdn_input_proj_conv_record(hidden, weights.query_key_value_z, conv_weight, conv_states,
                                    valid_columns, initial_slots, conv_record, query, key, value,
                                    output_gate_view, leaf_workspace, stream);
}

void Variant::gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                    qwen3_6::TextPhase, WorkspaceArena& workspace,
                                    cudaStream_t stream) {
    ops::linear_add(hidden, weight, residual, workspace, stream);
}

void Variant::gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const GdnProjectionWeights& weights,
                                          Tensor& hidden, Tensor& g, Tensor& beta,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    ops::gdn_norm_gating_proj(residual, norm_weight, eps, weights.a_b_projection, weights.a_log,
                              weights.dt_bias, workspace, hidden, g, beta, stream);
}

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         qwen3_6::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    run_sparse_moe(hidden, weights.op, residual, workspace, stream);
}

void Variant::mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                             Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream) {
    run_sparse_moe(hidden, weights.op, residual, workspace, stream);
}

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(std::int32_t first,
                                                                       std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(std::int32_t first,
                                                                std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::query_size, last});
    (void)layout.alloc(DType::BF16, {TextConfig::query_size, last});
    return layout.peak_bytes(1);
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::kv_size, last});
    (void)layout.alloc(DType::BF16, {TextConfig::kv_size, last});
    return layout.peak_bytes(1);
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(WeightsProfile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    return ops::attn_input_proj_workspace_capacity_bytes(
        QType::W8G32_F16S, 9216, TextConfig::hidden, ops::LinearPolicy::A16Only, first, last);
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(WeightsProfile,
                                                                          qwen3_6::TextPhase,
                                                                          std::int32_t first,
                                                                          std::int32_t last) {
    return ops::linear_add_workspace_capacity_bytes(QType::W8G32_F16S, TextConfig::hidden,
                                                    TextConfig::query_size,
                                                    ops::LinearPolicy::A16Only, first, last);
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(WeightsProfile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    return ops::gdn_input_proj_workspace_capacity_bytes(
        QType::W8G32_F16S, 12288, TextConfig::hidden, ops::LinearPolicy::A16Only, first, last);
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(WeightsProfile,
                                                                            qwen3_6::TextPhase,
                                                                            std::int32_t batch_size,
                                                                            std::int32_t first,
                                                                            std::int32_t last) {
    return ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch_size, first, last);
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(WeightsProfile,
                                                                          qwen3_6::TextPhase,
                                                                          std::int32_t batch_size,
                                                                          std::int32_t first,
                                                                          std::int32_t last) {
    return std::max(kMinimumLeafWorkspaceBytes,
                    ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                        TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch_size,
                        first, last));
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(WeightsProfile,
                                                                    qwen3_6::TextPhase,
                                                                    std::int32_t first,
                                                                    std::int32_t last) {
    return ops::linear_add_workspace_capacity_bytes(QType::W8G32_F16S, TextConfig::hidden,
                                                    TextConfig::value_dim,
                                                    ops::LinearPolicy::A16Only, first, last);
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t first,
                                                                          std::int32_t last) {
    return ops::gdn_norm_gating_proj_workspace_capacity_bytes(TextConfig::gdn_value_heads,
                                                              TextConfig::hidden, first, last);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(WeightsProfile, qwen3_6::TextPhase,
                                                         std::int32_t first, std::int32_t last) {
    return std::max(
        ops::sparse_moe_workspace_capacity_bytes(QType::Q4G64_F16S, QType::Q5G64_F16S, first, last),
        ops::sparse_moe_workspace_capacity_bytes(QType::Q4G64_F16S, QType::Q6G64_F16S, first,
                                                 last));
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(std::int32_t first,
                                                             std::int32_t last) {
    return ops::sparse_moe_workspace_capacity_bytes(QType::W8G32_F16S, QType::W8G32_F16S, first,
                                                    last);
}

} // namespace ninfer::targets::qwen3_6_35b_a3b::detail
