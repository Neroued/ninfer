#include "targets/qwen3_6_27b/impl/variant.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"

#include <algorithm>
#include <stdexcept>

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/instantiate.h"

namespace ninfer::targets::qwen3_6_27b::detail {
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

} // namespace

std::vector<GraphFrontierRange> Variant::ordinary_graph_ranges(std::uint32_t capacity) {
    // E+1 is the one-token visible window. Early ranges limit empty producer CTAs; later ranges
    // follow measured split-policy transitions until the producer grid reaches its fixed cap.
    return graph_ranges_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
}

std::vector<GraphFrontierRange> Variant::mtp_graph_ranges(std::uint32_t capacity,
                                                          std::uint32_t draft_window) {
    if (draft_window == 0 || 2ULL * draft_window > capacity) { return {}; }
    // Bound the final AR window E+2K at split-policy transitions until the grid reaches its cap.
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, 2 * draft_window);
    }
    // Target verify and MTP batch both have T=K+1 and W=E+K+1. Preserve one concrete INT8
    // implementation per range at the T=4/5/6 launch boundaries.
    if (draft_window == 3) {
        add_shifted(1029, draft_window + 1);
    } else if (draft_window == 4) {
        for (const std::uint32_t visible_end : {128U, 512U, 1029U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    } else if (draft_window == 5) {
        for (const std::uint32_t visible_end : {128U, 160U, 2054U, 8198U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_ranges_through(capacity - 2 * draft_window, ends);
}

std::vector<GraphFrontierRange> Variant::dflash_graph_ranges(std::uint32_t, std::uint32_t) {
    return {};
}

void Variant::attach_diagnostics(qwen3_6::Program<Variant>& program, void* context,
                                 qwen3_6::TextTapCallback text, qwen3_6::VisionTapCallback vision) {
    program.impl_->diagnostic_context    = context;
    program.impl_->diagnostic_text_tap   = text;
    program.impl_->diagnostic_vision_tap = vision;
}

void Variant::detach_diagnostics(qwen3_6::Program<Variant>& program) noexcept {
    program.impl_->diagnostic_context    = nullptr;
    program.impl_->diagnostic_text_tap   = nullptr;
    program.impl_->diagnostic_vision_tap = nullptr;
}

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    ops::attn_input_proj(hidden, weights.query_key, weights.gate_value, query, gate, key, value,
                         workspace, stream);
}

void Variant::mtp_attention_projection(const Tensor& hidden,
                                       const MtpAttentionProjectionWeights& weights, Tensor& query,
                                       Tensor& gate, Tensor& key, Tensor& value,
                                       WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor packed  = workspace.alloc(DType::BF16, {TextConfig::mtp_attention_input_rows, cols});
    ops::linear(hidden, weights.packed, packed, workspace, stream);
    Tensor query_heads = query.view({TextConfig::head_dim, TextConfig::query_heads, cols});
    Tensor key_heads   = key.view({TextConfig::head_dim, TextConfig::kv_heads, cols});
    Tensor gate_heads  = gate.view({TextConfig::head_dim, TextConfig::query_heads, cols});
    Tensor value_heads = value.view({TextConfig::head_dim, TextConfig::kv_heads, cols});
    ops::mtp_split_attn_in(packed, query_heads, key_heads, gate_heads, value_heads, stream);
}

void Variant::mtp_kv_projection(const Tensor& hidden, const MtpAttentionProjectionWeights& weights,
                                Tensor& key, Tensor& value, WorkspaceArena& workspace,
                                cudaStream_t stream) {
    ops::linear_pair(hidden, weights.key, weights.value, key, value, workspace, stream);
}

void Variant::mtp_q_gate_projection(const Tensor& hidden,
                                    const MtpAttentionProjectionWeights& weights, Tensor& query,
                                    Tensor& gate, WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear(hidden, weights.query, query, workspace, stream);
    ops::linear(hidden, weights.output_gate, gate, workspace, stream);
}

void Variant::gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                   Tensor& qkv, Tensor& output_gate, WorkspaceArena& workspace,
                                   cudaStream_t stream) {
    (void)output_gate;
    ops::gdn_input_proj(hidden, weights.query_key, weights.value, qkv, workspace, stream);
}

void Variant::gdn_input_projection_snapshot(const Tensor& hidden,
                                            const GdnProjectionWeights& weights,
                                            const Tensor& conv_weight, Tensor& conv_states,
                                            const Tensor& initial_slot, Tensor& query, Tensor& key,
                                            Tensor& value, Tensor& output_gate,
                                            WorkspaceArena& workspace, cudaStream_t stream) {
    (void)output_gate;
    ops::gdn_input_proj_conv_snapshot(hidden, weights.query_key, weights.value, conv_weight,
                                      conv_states, initial_slot, query, key, value, workspace,
                                      stream);
}

void Variant::gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const GdnProjectionWeights& weights,
                                          Tensor& hidden, Tensor& g, Tensor& beta,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    ops::gdn_norm_gating_proj(residual, norm_weight, eps, weights.a_projection,
                              weights.b_projection, weights.a_log, weights.dt_bias, workspace,
                              hidden, g, beta, stream);
}

void Variant::gdn_output_gate_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                         Tensor& output_gate, WorkspaceArena& workspace,
                                         cudaStream_t stream) {
    Tensor output_gate_flat =
        output_gate.view({TextConfig::value_dim, static_cast<int>(hidden.ne[1])});
    ops::linear(hidden, weights.z, output_gate_flat, workspace, stream);
}

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope        = workspace.scope();
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, hidden.ne[1]});
    ops::linear_swiglu(hidden, weights.gate_up, activation, workspace, stream);
    ops::linear_add(activation, weights.down, residual, workspace, stream);
}

void Variant::mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                             Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream) {
    const int cols = hidden.ne[1];
    Tensor gate_up = workspace.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, cols});
    ops::linear(hidden, weights.gate_up, gate_up, workspace, stream);
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, cols});
    ops::silu_mul(gate_up.slice(0, 0, TextConfig::intermediate),
                  gate_up.slice(0, TextConfig::intermediate, TextConfig::intermediate), activation,
                  stream);
    Tensor delta = workspace.alloc(DType::BF16, {TextConfig::hidden, cols});
    ops::linear(activation, weights.down, delta, workspace, stream);
    ops::residual_add(delta, residual, stream);
}

std::size_t Variant::mtp_attention_workspace_bytes(std::int32_t tokens) {
    return static_cast<std::size_t>(TextConfig::mtp_attention_input_rows) * tokens *
           dtype_size(DType::BF16);
}

std::size_t Variant::mtp_kv_workspace_bytes(std::int32_t) { return 0; }

std::size_t Variant::mtp_q_gate_workspace_bytes(std::int32_t) { return 0; }

std::size_t Variant::gdn_input_projection_workspace_bytes(std::int32_t tokens) {
    return ops::gdn_input_proj_workspace_bytes(2 * TextConfig::key_dim, TextConfig::value_dim,
                                               tokens);
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_bytes(std::int32_t tokens) {
    return ops::gdn_input_proj_conv_snapshot_workspace_bytes(
        TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, tokens);
}

std::size_t Variant::gdn_norm_control_projection_workspace_bytes(std::int32_t tokens) {
    return ops::gdn_norm_gating_proj_workspace_bytes(tokens);
}

std::size_t Variant::gdn_output_gate_projection_workspace_bytes(std::int32_t) { return 0; }

std::size_t Variant::post_mixer_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, tokens});
    {
        auto scope = layout.scope();
        layout.alloc_bytes(
            ops::linear_swiglu_workspace_bytes(2 * TextConfig::intermediate, tokens));
    }
    {
        auto scope = layout.scope();
        layout.alloc_bytes(
            ops::linear_add_workspace_bytes(TextConfig::hidden, TextConfig::intermediate, tokens));
    }
    return layout.peak_bytes();
}

std::size_t Variant::mtp_post_mixer_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, tokens});
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, tokens});
    (void)layout.alloc(DType::BF16, {TextConfig::hidden, tokens});
    return layout.peak_bytes();
}

} // namespace ninfer::targets::qwen3_6_27b::detail
