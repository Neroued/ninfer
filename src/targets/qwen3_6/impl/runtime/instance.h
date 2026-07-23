#pragma once

#ifndef NINFER_QWEN36_VARIANT
#    error "NINFER_QWEN36_VARIANT must name the complete exact Variant"
#endif
#ifndef NINFER_QWEN36_RUNTIME_NS
#    error "NINFER_QWEN36_RUNTIME_NS must be a unique identifier for this instantiation"
#endif

#include <ninfer/targets/qwen3_6/runtime.h>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using Variant                        = NINFER_QWEN36_VARIANT;
using TextConfig                     = typename Variant::TextConfig;
using VisionConfig                   = typename Variant::VisionConfig;
using LoadedModelData                = typename Variant::ModelView;
using FullAttentionWeights           = typename LoadedModelData::FullLayer;
using GdnWeights                     = typename LoadedModelData::GdnLayer;
using MlpWeights                     = typename Variant::PostMixerWeights;
using MtpWeights                     = typename LoadedModelData::MtpLayer;
using FullAttentionProjectionWeights = typename Variant::FullAttentionProjectionWeights;
using GdnProjectionWeights           = typename Variant::GdnProjectionWeights;
using VisionWeights                  = typename Variant::VisionWeights;
using GraphFrontierRange             = typename Variant::GraphFrontierRange;

using SequencePlan = qwen3_6::SequencePlan<Variant>;
using RequestPlan  = qwen3_6::RequestPlan<Variant>;
using Program      = qwen3_6::Program<Variant>;

inline constexpr float kAttentionScale                = Variant::attention_scale;
inline constexpr float kGdnScale                      = Variant::gdn_scale;
inline constexpr std::uint32_t kPrefillChunkAlignment = Variant::prefill_chunk_alignment;
inline constexpr std::uint32_t kMaximumMtpDraftTokens = Variant::maximum_mtp_draft_tokens;

inline std::vector<GraphFrontierRange> ordinary_graph_ranges(std::uint32_t capacity) {
    return Variant::ordinary_graph_ranges(capacity);
}

inline std::vector<GraphFrontierRange> mtp_graph_ranges(std::uint32_t capacity,
                                                        std::uint32_t draft_window) {
    return Variant::mtp_graph_ranges(capacity, draft_window);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
