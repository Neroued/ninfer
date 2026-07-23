#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/sampling.h"
#include "core/decode_graph.h"
#include "runtime/contract/transient_region.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include "core/kv_cache.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/vision_prefill.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <functional>
#include <optional>
#include <span>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

using qwen3_6::PreparedPromptData;
using qwen3_6::PromptModality;

struct State {
    DeviceContext& device;
    const LoadedModelData& model;
    WorkspaceArena& work;
    KVCache& text_kv;
    KVCache* mtp_kv;
    qwen3_6::GdnStateStore& gdn;
    qwen3_6::RoundState& io;
    Tensor& prefill_hidden;
    std::uint32_t prefill_chunk;
    std::uint32_t text_kv_base;
    const ops::SamplingConfig* sampling;
    ProposalHead proposal_head;
    Tensor* tail_hidden;
    Tensor* boundary_hidden;
    void* diagnostic_context                = nullptr;
    TextTapCallback diagnostic_text_tap     = nullptr;
    VisionTapCallback diagnostic_vision_tap = nullptr;
};

struct MtpGqaEnvelopes {
    ops::GqaExecutionEnvelope target_verify;
    ops::GqaExecutionEnvelope batch;
    std::array<ops::GqaExecutionEnvelope, kMaximumMtpDraftTokens - 1> ar;
};

using GraphPrepare = std::function<void()>;

void configure_text_card(TextContext& card, const State& state);

[[nodiscard]] bool prefill_text(State& state, std::span<const TokenId> ids,
                                std::optional<std::uint32_t> snapshot_boundary, bool prepare_mtp);

struct MultimodalPrefillResult {
    bool mtp_prepared                = false;
    std::uint32_t final_chunk_tokens = 0;
    double vision_seconds            = 0.0;
};

[[nodiscard]] MultimodalPrefillResult
prefill_multimodal(State& state, const PreparedPromptData& prompt, const VisionPrefillPlan& plan,
                   runtime::TransientRegion transient,
                   std::optional<std::uint32_t> snapshot_boundary, bool prepare_mtp);

void sample_from_hidden(State& state, const Tensor& hidden, std::int32_t absolute_position,
                        std::int32_t purpose);
void target_verify(TextContext& card, State& state, const Tensor& ids, const Tensor& positions,
                   ops::GqaExecutionEnvelope envelope);
void speculative_verify_and_accept(State& state, TextContext& card, std::uint32_t draft_window,
                                   ops::GqaExecutionEnvelope target_envelope);
void mtp_bridge_and_propose(State& state, const Tensor& next_token, const Tensor& previous_hidden,
                            std::int32_t position, std::span<const std::int32_t> rope_position,
                            bool build_proposal);

// Executes an exact one-token target step through the verify schedule. The resulting target hidden
// is in io.verify_hidden[:,0], the sampled token is in io.token, and GDN snapshot slot 0 is the
// resulting recurrent state.
void warm_capture_ordinary_round(State& state, bool align_mtp, ops::GqaExecutionEnvelope envelope,
                                 const GraphPrepare& prepare, DecodeGraph& graph);
void ordinary_round(State& state, bool align_mtp, ops::GqaExecutionEnvelope envelope,
                    DecodeGraph* graph);

// Executes one fixed-k MTP synchronization/proposal round around the common speculative
// verification transaction. The number of licensed tokens and their values are written to
// io.speculative.
void warm_capture_mtp_round(State& state, std::uint32_t k, MtpGqaEnvelopes envelopes,
                            const GraphPrepare& prepare, DecodeGraph& graph);
void mtp_round(State& state, std::uint32_t k, MtpGqaEnvelopes envelopes, DecodeGraph* graph);

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
