#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/sampling.h"
#include "core/decode_graph.h"
#include "runtime/contract/transient_region.h"
#include "targets/qwen3_6_27b_rtx5090/impl/frontend/frontend.h"
#include "targets/qwen3_6_27b_rtx5090/impl/config.h"
#include "targets/qwen3_6_27b_rtx5090/impl/load/bindings.h"
#include "core/kv_cache.h"
#include "targets/qwen3_6_27b_rtx5090/impl/state/state_store.h"
#include "targets/qwen3_6_27b_rtx5090/impl/schedule/text_context.h"
#include "targets/qwen3_6_27b_rtx5090/impl/schedule/vision_context.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <functional>
#include <optional>
#include <span>

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail::schedule {

struct State {
    DeviceContext& device;
    const LoadedModelData& model;
    WorkspaceArena& work;
    KVCache& text_kv;
    KVCache* mtp_kv;
    GdnState& gdn;
    StepState& io;
    std::uint32_t prefill_chunk;
    std::uint32_t text_kv_base;
    const ops::SamplingConfig* sampling;
    ProposalHead proposal_head;
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

[[nodiscard]] std::size_t vision_workspace_bytes(const PreparedPromptData& prompt);
[[nodiscard]] ProcessedInput take_processed_prompt(PreparedPromptData&& prompt);
[[nodiscard]] Tensor encode_vision(State& state, const ProcessedInput& prompt,
                                   runtime::TransientRegion transient);

[[nodiscard]] bool prefill_text(State& state, std::span<const TokenId> ids,
                                std::optional<std::uint32_t> snapshot_boundary, bool prepare_mtp);
[[nodiscard]] bool prefill_multimodal(State& state, const ProcessedInput& prompt,
                                      const Tensor& visual_embeddings, bool prepare_mtp);

void sample_from_hidden(State& state, const Tensor& hidden, std::int32_t absolute_position,
                        std::int32_t purpose);
void mtp_bridge_and_propose(State& state, const Tensor& next_token, const Tensor& previous_hidden,
                            std::int32_t position, bool build_proposal);

// Executes an exact one-token target step through the verify schedule. The resulting target hidden
// is in io.verify_hidden[:,0], the sampled token is in io.token, and GDN snapshot slot 0 is the
// resulting recurrent state.
void warm_capture_ordinary_round(State& state, bool align_mtp, ops::GqaExecutionEnvelope envelope,
                                 const GraphPrepare& prepare, DecodeGraph& graph);
void ordinary_round(State& state, bool align_mtp, ops::GqaExecutionEnvelope envelope,
                    DecodeGraph* graph);

// Executes one fixed-k verify/accept/propose round. The number of licensed tokens is written to
// io.num_sampled and the tokens to io.sampled_out.
void warm_capture_mtp_round(State& state, std::uint32_t k, MtpGqaEnvelopes envelopes,
                            const GraphPrepare& prepare, DecodeGraph& graph);
void mtp_round(State& state, std::uint32_t k, MtpGqaEnvelopes envelopes, DecodeGraph* graph);

} // namespace ninfer::targets::qwen3_6_27b_rtx5090::detail::schedule
