#pragma once

#include "core/buffer.h"
#include "core/layout.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/types.h"

#include <cstdint>
#include <optional>
#include <span>

namespace ninfer::models::qwen3_5 {

struct FrameSpec {
    std::int32_t hidden               = 0;
    std::int32_t output_rows          = 0;
    std::int32_t public_tokens        = 0;
    std::uint32_t prefill_chunk       = 0;
    std::uint32_t batch_capacity      = 1;
    std::uint32_t draft_window        = 0;
    SpeculativeBackend backend        = SpeculativeBackend::None;
    bool causal_scoring               = false;
    bool vision                       = false;
    std::uint32_t control_token_limit = 0;
};

struct FrameIngressLayout {
    TensorRegion live_rows;
    TensorRegion anchors;
    TensorRegion frontiers;
    TensorRegion backend_frontiers;
    TensorRegion extents;
    TensorRegion valid_columns;
    TensorRegion proposal_valid_columns;
    TensorRegion budgets;
    TensorRegion main_table_rows;
    TensorRegion backend_table_rows;
    TensorRegion residency_slots;
    TensorRegion rope_deltas;
    TensorRegion sampling_positions;
    TensorRegion history_skip;
    TensorRegion previous_rope;
    TensorRegion decode_positions;
    TensorRegion decode_rope;
    TensorRegion step_positions;
    TensorRegion step_valid;
    TensorRegion step_rope;
    TensorRegion state_source_offsets;
    TensorRegion state_destination_offsets;
    LayoutRegion sampling;
    std::size_t bytes = 0;
};

struct FrameCommitLayout {
    TensorRegion state_destination_offsets;
    TensorRegion last_columns;
    TensorRegion accepted_columns;
    TensorRegion count_columns;
    TensorRegion draft_append_starts;
    TensorRegion draft_append_ends;
    std::size_t bytes = 0;
};

struct FrameEgressLayout {
    TensorRegion tokens, counts;
    std::size_t bytes = 0;
};

struct FrameControlLayout {
    FrameIngressLayout ingress;
    FrameCommitLayout commit;
    FrameEgressLayout egress;
};

struct FrameIngressView {
    std::span<std::int32_t> live_rows;
    std::span<std::int32_t> anchors;
    std::span<std::int32_t> frontiers;
    std::span<std::int32_t> backend_frontiers;
    std::span<std::int32_t> extents;
    std::span<std::int32_t> valid_columns;
    std::span<std::int32_t> proposal_valid_columns;
    std::span<std::int32_t> budgets;
    std::span<std::int32_t> main_table_rows;
    std::span<std::int32_t> backend_table_rows;
    std::span<std::int32_t> residency_slots;
    std::span<std::int32_t> rope_deltas;
    std::span<std::int32_t> sampling_positions;
    std::span<std::int32_t> history_skip;
    std::span<std::int32_t> previous_rope;
    std::span<std::int32_t> decode_positions;
    std::span<std::int32_t> decode_rope;
    std::span<std::int32_t> step_positions;
    std::span<std::int32_t> step_valid;
    std::span<std::int32_t> step_rope;
    std::span<std::int64_t> state_source_offsets;
    std::span<std::int64_t> state_destination_offsets;
    std::span<ops::SamplingConfig> sampling;
};

struct FrameCommitView {
    std::span<std::int64_t> state_destination_offsets;
    std::span<std::int32_t> last_columns;
    std::span<std::int32_t> accepted_columns;
    std::span<std::int32_t> count_columns;
    std::span<std::int32_t> draft_append_starts;
    std::span<std::int32_t> draft_append_ends;
};

struct FrameEgressView {
    std::span<TokenId> tokens;
    std::span<std::int32_t> counts;
};

[[nodiscard]] FrameControlLayout plan_frame_controls(const FrameSpec&, std::uint32_t rows);

struct MtpFrameLayout {
    TensorRegion base_conditioning;
    TensorRegion ar_hidden;
    TensorRegion next_hidden;
    TensorRegion drafts;
    TensorRegion proposal_logits;
    TensorRegion history_conditioning;
    TensorRegion history_ids;
    TensorRegion history_positions, history_columns;
    TensorRegion history_rope_positions;
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
    std::optional<TensorRegion> score_logprobs;
    std::size_t bytes = 0;
};

struct FrameLayout {
    FrameSpec spec;
    FrameControlLayout controls;
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
    std::optional<TensorRegion> control_ids;
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
    Tensor history_conditioning, history_ids, history_positions, history_rope_positions,
        history_columns;
    Tensor step_positions, step_rope_positions, step_valid_columns;

    MtpFrame(MemorySpan backing, const MtpFrameLayout& layout);
    [[nodiscard]] MtpStepView step_view(std::int32_t step, std::int32_t batch) const;
    [[nodiscard]] Tensor proposals(std::int32_t batch) const;
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
    [[nodiscard]] FrameInputView decode_view(std::int32_t width, std::int32_t batch) const;
    [[nodiscard]] FrameInputView host_decode_view(std::int32_t width, std::int32_t batch) const;

    void bind_rows(std::uint32_t rows);
    void clear_ingress();
    void clear_commit();
    void clear_egress();

    [[nodiscard]] std::uint32_t rows() const noexcept { return rows_; }

    const FrameSpec spec;
    const std::int32_t input_capacity;
    PinnedHostBuffer host_storage;
    FrameIngressView host_ingress;
    FrameCommitView host_commit;
    FrameEgressView host_egress;
    MemorySpan host_ingress_bytes, host_commit_bytes, host_egress_bytes;
    Tensor host_input_ids, host_cache_positions, host_rope_positions;
    Tensor host_visual_indices;
    Tensor host_control_tokens;
    Tensor control_ids;
    Tensor host_masks, host_drafts, host_score_logprobs;
    Tensor host_step_positions, host_step_rope_positions, host_step_valid_columns;
    MemorySpan ingress, commit, egress;
    Tensor live_rows, anchors, frontiers, backend_frontiers, extents, valid_columns;
    Tensor proposal_valid_columns, budgets, main_table_rows, backend_table_rows;
    Tensor state_source_offsets, state_destination_offsets, residency_slots, rope_deltas;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor sampling_positions, history_skip, previous_rope;
    Tensor decode_positions, decode_rope;
    Tensor commit_destination_offsets, last_columns, count_columns, accepted_columns;
    Tensor draft_append_starts, draft_append_ends;
    Tensor output_tokens, output_counts;
    Tensor input_ids, cache_positions, rope_positions, hidden;
    Tensor input_embeddings;
    Tensor visual_indices;
    Tensor logits, target_argmax, masks, score_targets, score_logprobs;
    std::optional<MtpFrame> mtp;
    std::optional<DFlashFrame> dflash;

private:
    MemorySpan ingress_storage_, commit_storage_, egress_storage_;
    MemorySpan host_ingress_storage_, host_commit_storage_, host_egress_storage_;
    std::uint32_t rows_ = 0;
    void validate_input_shape(std::int32_t width, std::int32_t batch) const;
};

} // namespace ninfer::models::qwen3_5
