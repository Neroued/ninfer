#pragma once

#include "core/buffer.h"
#include "core/layout.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/types.h"

#include <array>
#include <cstdint>
#include <optional>

namespace ninfer::models::qwen3_5 {

struct FrameSpec {
    std::int32_t hidden          = 0;
    std::int32_t output_rows     = 0;
    std::int32_t public_tokens   = 0;
    std::uint32_t prefill_chunk  = 0;
    std::uint32_t batch_capacity = 1;
    std::uint32_t draft_window   = 0;
    SpeculativeBackend backend   = SpeculativeBackend::None;
    bool causal_scoring          = false;
    bool vision                  = false;
};

// Arrays are compact-row indexed. Tensor views bind only the current exact-B prefix.
struct FrameIngress {
    std::array<TokenId, kMaximumConcurrency> anchors{};
    std::array<std::int32_t, kMaximumConcurrency> frontiers{};
    std::array<std::int32_t, kMaximumConcurrency> backend_frontiers{};
    std::array<std::int32_t, kMaximumConcurrency> extents{};
    std::array<std::int32_t, kMaximumConcurrency> valid_columns{};
    std::array<std::int32_t, kMaximumConcurrency> proposal_valid_columns{};
    std::array<std::int32_t, kMaximumConcurrency> budgets{};
    std::array<std::int32_t, kMaximumConcurrency> main_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> backend_table_rows{};
    std::array<std::int64_t, kMaximumConcurrency> state_source_offsets{};
    std::array<std::int64_t, kMaximumConcurrency> state_destination_offsets{};
    std::array<std::int32_t, kMaximumConcurrency> residency_slots{};
    std::array<std::int32_t, kMaximumConcurrency> rope_deltas{};
    std::array<ops::SamplingConfig, kMaximumConcurrency> sampling{};
};

struct FrameCommit {
    std::array<std::int64_t, kMaximumConcurrency> state_destination_offsets{};
    std::array<std::int32_t, kMaximumConcurrency> history_columns{};
    std::array<std::int32_t, kMaximumConcurrency> draft_append_starts{};
    std::array<std::int32_t, kMaximumConcurrency> draft_append_ends{};
};

struct FrameEgress {
    std::array<TokenId, kMaximumConcurrency * 16> tokens{};
    std::array<std::int32_t, kMaximumConcurrency> counts{};
    std::array<std::int32_t, kMaximumConcurrency> accepted_drafts{};
};

struct MtpFrameLayout {
    TensorRegion base_conditioning;
    TensorRegion ar_hidden;
    TensorRegion next_hidden;
    TensorRegion drafts;
    TensorRegion proposal_logits;
    TensorRegion history_conditioning;
    TensorRegion history_ids;
    TensorRegion history_positions;
    TensorRegion history_rope_positions;
    TensorRegion step_ids;
    TensorRegion step_positions;
    TensorRegion step_rope_positions;
    TensorRegion step_valid_columns;
};

struct DFlashFrameLayout {
    TensorRegion proposal_ids;
    TensorRegion proposal_positions;
    TensorRegion append_positions;
    TensorRegion append_counts;
    TensorRegion drafts;
    std::optional<TensorRegion> candidate_ids;
    std::optional<TensorRegion> proposal_q;
};

struct FrameHostLayout {
    LayoutRegion ingress;
    LayoutRegion commit;
    LayoutRegion egress;
    TensorRegion input_ids;
    TensorRegion cache_positions;
    TensorRegion rope_positions;
    std::optional<TensorRegion> visual_indices;
    std::optional<TensorRegion> control_tokens;
    std::optional<TensorRegion> masks;
    std::optional<TensorRegion> drafts;
    std::optional<TensorRegion> step_positions;
    std::optional<TensorRegion> step_rope_positions;
    std::optional<TensorRegion> step_valid_columns;
    std::optional<TensorRegion> score_logprobs;
    std::size_t bytes = 0;
};

struct FrameLayout {
    FrameSpec spec;
    std::int32_t input_capacity = 0;
    LayoutRegion ingress;
    LayoutRegion commit;
    LayoutRegion egress;
    TensorRegion input_ids;
    TensorRegion cache_positions;
    TensorRegion rope_positions;
    TensorRegion hidden;
    std::optional<TensorRegion> input_embeddings;
    std::optional<TensorRegion> visual_indices;
    std::optional<TensorRegion> logits;
    std::optional<TensorRegion> target_argmax;
    std::optional<TensorRegion> masks;
    std::optional<TensorRegion> score_targets;
    std::optional<TensorRegion> score_logprobs;
    std::optional<MtpFrameLayout> mtp;
    std::optional<DFlashFrameLayout> dflash;
    FrameHostLayout host;
};

[[nodiscard]] FrameLayout plan_frames(LayoutBuilder& builder, const FrameSpec& spec);

struct MtpStepView {
    Tensor ids, positions, rope_positions, valid_columns;
};

struct MtpFrame {
    Tensor base_conditioning, ar_hidden, next_hidden, drafts, proposal_logits;
    Tensor history_conditioning, history_ids, history_positions, history_rope_positions;
    Tensor step_ids, step_positions, step_rope_positions, step_valid_columns;

    MtpFrame(MemorySpan backing, const MtpFrameLayout& layout);
    [[nodiscard]] MtpStepView step_view(std::int32_t step, std::int32_t batch) const;
};

struct DFlashFrame {
    Tensor proposal_ids, proposal_positions, append_positions, append_counts, drafts;
    Tensor candidate_ids, proposal_q;

    DFlashFrame(MemorySpan backing, const DFlashFrameLayout& layout);
};

struct FrameInputView {
    Tensor ids, cache_positions, rope_positions, hidden;
};

// One stable device frame and its pinned transfer storage. The execution ticket retains both
// until all device, transfer and control-side consumers finish.
class ExecutionFrame {
public:
    ExecutionFrame(MemorySpan backing, const FrameLayout& layout);
    ExecutionFrame(const ExecutionFrame&)            = delete;
    ExecutionFrame& operator=(const ExecutionFrame&) = delete;

    [[nodiscard]] FrameInputView input_view(std::int32_t width, std::int32_t batch) const;
    [[nodiscard]] FrameInputView host_input_view(std::int32_t width, std::int32_t batch) const;

    const FrameSpec spec;
    const std::int32_t input_capacity;
    PinnedHostBuffer host_storage;
    FrameIngress* host_ingress = nullptr;
    FrameCommit* host_commit   = nullptr;
    FrameEgress* host_egress   = nullptr;
    Tensor host_input_ids, host_cache_positions, host_rope_positions;
    Tensor host_visual_indices;
    Tensor host_control_tokens;
    Tensor host_masks, host_drafts, host_score_logprobs;
    Tensor host_step_positions, host_step_rope_positions, host_step_valid_columns;
    MemorySpan ingress, commit, egress;
    Tensor anchors, frontiers, backend_frontiers, extents, valid_columns;
    Tensor proposal_valid_columns, budgets, main_table_rows, backend_table_rows;
    Tensor state_source_offsets, state_destination_offsets, residency_slots, rope_deltas;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor commit_destination_offsets, history_columns;
    Tensor draft_append_starts, draft_append_ends;
    Tensor output_tokens, output_counts, accepted_drafts;
    Tensor input_ids, cache_positions, rope_positions, hidden;
    Tensor input_embeddings;
    Tensor visual_indices;
    Tensor logits, target_argmax, masks, score_targets, score_logprobs;
    std::optional<MtpFrame> mtp;
    std::optional<DFlashFrame> dflash;

private:
    void validate_input_shape(std::int32_t width, std::int32_t batch) const;
};

} // namespace ninfer::models::qwen3_5
