#pragma once

#include "targets/qwen3_6_35b_a3b/impl/config.h"
#include "targets/qwen3_6_35b_a3b/impl/load/bindings.h"
#include <ninfer/targets/qwen3_6/runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::targets::qwen3_6_35b_a3b::detail {

using GraphFrontierRange = qwen3_6::GraphFrontierRange;

struct Variant {
    using TextConfig                     = detail::TextConfig;
    using VisionConfig                   = detail::VisionConfig;
    using DFlashConfig                   = detail::DFlashConfig;
    using ModelView                      = detail::RuntimeModelView;
    using FullAttentionProjectionWeights = detail::AttentionProjectionPayload;
    using GdnProjectionWeights           = detail::GdnProjectionPayload;
    using PostMixerWeights               = detail::SparseMoePayload;
    using MtpAttentionProjectionWeights  = detail::AttentionProjectionPayload;
    using MtpPostMixerWeights            = detail::SparseMoePayload;
    using VisionWeights                  = qwen3_6::VisionWeights;
    using GraphFrontierRange             = detail::GraphFrontierRange;

    static constexpr float attention_scale                     = kAttentionScale;
    static constexpr float gdn_scale                           = kGdnScale;
    static constexpr std::uint32_t prefill_chunk_alignment     = kPrefillChunkAlignment;
    static constexpr std::uint32_t maximum_mtp_draft_tokens    = kMaximumMtpDraftTokens;
    static constexpr std::uint32_t maximum_dflash_draft_tokens = kMaximumDFlashDraftTokens;
    static constexpr std::uint32_t maximum_context             = kNativeContext;
    static constexpr bool supports_dflash                      = DFlashConfig::supported;
    static constexpr std::int32_t draft_head_rows              = 131072;

    [[nodiscard]] static std::vector<GraphFrontierRange>
    ordinary_graph_ranges(std::uint32_t capacity);
    [[nodiscard]] static std::vector<GraphFrontierRange>
    mtp_graph_ranges(std::uint32_t capacity, std::uint32_t draft_window);
    [[nodiscard]] static std::vector<GraphFrontierRange>
    dflash_graph_ranges(std::uint32_t capacity, std::uint32_t draft_window);

    static void attention_projection(const Tensor& hidden,
                                     const FullAttentionProjectionWeights& weights, Tensor& query,
                                     Tensor& gate, Tensor& key, Tensor& value,
                                     WorkspaceArena& workspace, cudaStream_t stream);
    static void mtp_attention_projection(const Tensor& hidden,
                                         const MtpAttentionProjectionWeights& weights,
                                         Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                                         WorkspaceArena& workspace, cudaStream_t stream);
    static void mtp_kv_projection(const Tensor& hidden,
                                  const MtpAttentionProjectionWeights& weights, Tensor& key,
                                  Tensor& value, WorkspaceArena& workspace, cudaStream_t stream);
    static void mtp_q_gate_projection(const Tensor& hidden,
                                      const MtpAttentionProjectionWeights& weights, Tensor& query,
                                      Tensor& gate, WorkspaceArena& workspace, cudaStream_t stream);
    static void gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                     Tensor& qkv, Tensor& output_gate, WorkspaceArena& workspace,
                                     cudaStream_t stream);
    static void gdn_input_projection_snapshot(const Tensor& hidden,
                                              const GdnProjectionWeights& weights,
                                              const Tensor& conv_weight, Tensor& conv_states,
                                              const Tensor& initial_slot, Tensor& query,
                                              Tensor& key, Tensor& value, Tensor& output_gate,
                                              WorkspaceArena& workspace, cudaStream_t stream);
    static void gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                            float eps, const GdnProjectionWeights& weights,
                                            Tensor& hidden, Tensor& g, Tensor& beta,
                                            WorkspaceArena& workspace, cudaStream_t stream);
    static void gdn_output_gate_projection(const Tensor& hidden,
                                           const GdnProjectionWeights& weights, Tensor& output_gate,
                                           WorkspaceArena& workspace, cudaStream_t stream);
    static void post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                           WorkspaceArena& workspace, cudaStream_t stream);
    static void mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                               Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream);

    [[nodiscard]] static std::size_t mtp_attention_workspace_bytes(std::int32_t tokens);
    [[nodiscard]] static std::size_t mtp_kv_workspace_bytes(std::int32_t tokens);
    [[nodiscard]] static std::size_t mtp_q_gate_workspace_bytes(std::int32_t tokens);
    [[nodiscard]] static std::size_t gdn_input_projection_workspace_bytes(std::int32_t tokens);
    [[nodiscard]] static std::size_t
    gdn_input_projection_snapshot_workspace_bytes(std::int32_t tokens);
    [[nodiscard]] static std::size_t
    gdn_norm_control_projection_workspace_bytes(std::int32_t tokens);
    [[nodiscard]] static std::size_t
    gdn_output_gate_projection_workspace_bytes(std::int32_t tokens);
    [[nodiscard]] static std::size_t post_mixer_workspace_bytes(std::int32_t tokens);
    [[nodiscard]] static std::size_t mtp_post_mixer_workspace_bytes(std::int32_t tokens);
};

} // namespace ninfer::targets::qwen3_6_35b_a3b::detail
