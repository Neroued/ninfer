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
using WeightsProfile                 = typename Variant::WeightsProfile;
using TextConfig                     = typename Variant::TextConfig;
using VisionConfig                   = typename Variant::VisionConfig;
using DFlashConfig                   = typename Variant::DFlashConfig;
using LoadedModelData                = typename Variant::ModelView;
using FullAttentionWeights           = typename LoadedModelData::FullLayer;
using GdnWeights                     = typename LoadedModelData::GdnLayer;
using MlpWeights                     = typename Variant::PostMixerWeights;
using MtpWeights                     = typename LoadedModelData::MtpLayer;
using DFlashWeights                  = typename LoadedModelData::DFlash;
using FullAttentionProjectionWeights = typename Variant::FullAttentionProjectionWeights;
using GdnProjectionWeights           = typename Variant::GdnProjectionWeights;
using VisionWeights                  = typename Variant::VisionWeights;
using GraphExecutionProfile          = typename Variant::GraphExecutionProfile;

using SequencePlan            = qwen3_6::SequencePlan<Variant>;
using SequencePlanner         = qwen3_6::SequencePlanner<Variant>;
using RequestBasePlan         = qwen3_6::RequestBasePlan<Variant>;
using AdmissionCandidate      = qwen3_6::AdmissionCandidate<Variant>;
using PressurePlanningSession = qwen3_6::PressurePlanningSession<Variant>;
using PressureTargetHandle    = qwen3_6::PressureTargetHandle;
using ResourcePlan            = qwen3_6::ResourcePlan<Variant>;
using PersistentBackfillProof = qwen3_6::PersistentBackfillProof<Variant>;
using SequenceHandle          = qwen3_6::SequenceHandle<Variant>;
using ContinuationHandle      = qwen3_6::ContinuationHandle<Variant>;
using SharedPrefixHandle      = qwen3_6::SharedPrefixHandle<Variant>;
using CaptureOffer            = qwen3_6::CaptureOffer<Variant>;
using CaptureAssessment       = qwen3_6::CaptureAssessment;
using ActiveCaptureResult     = qwen3_6::ActiveCaptureResult<Variant>;
using MaterializationResult   = qwen3_6::MaterializationResult<Variant>;
using PendingBatch            = qwen3_6::PendingBatch<Variant>;
using PrefillProgress         = qwen3_6::PrefillProgress<Variant>;
using StartResult             = qwen3_6::StartResult<Variant>;
using CommitResult            = qwen3_6::CommitResult<Variant>;
using DiscardResult           = qwen3_6::DiscardResult<Variant>;
using FinishResult            = qwen3_6::FinishResult<Variant>;
using AbortResult             = qwen3_6::AbortResult<Variant>;
using ReleaseResult           = qwen3_6::ReleaseResult<Variant>;
using ContractAccess          = qwen3_6::detail::RuntimeContractAccess<Variant>;
using Program                 = qwen3_6::Program<Variant>;

inline constexpr float kAttentionScale                   = Variant::attention_scale;
inline constexpr float kGdnScale                         = Variant::gdn_scale;
inline constexpr std::uint32_t kPrefillChunkAlignment    = Variant::prefill_chunk_alignment;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = Variant::maximum_mtp_draft_tokens;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = Variant::maximum_dflash_draft_tokens;

inline std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity) {
    return Variant::ordinary_graph_profiles(capacity);
}

inline std::vector<GraphExecutionProfile> mtp_graph_profiles(std::uint32_t capacity,
                                                             std::uint32_t draft_window) {
    return Variant::mtp_graph_profiles(capacity, draft_window);
}

inline std::vector<GraphExecutionProfile> dflash_graph_profiles(std::uint32_t capacity,
                                                                std::uint32_t draft_window,
                                                                std::uint32_t batch_size) {
    return Variant::dflash_graph_profiles(capacity, draft_window, batch_size);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
