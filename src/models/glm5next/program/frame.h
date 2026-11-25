#pragma once

#include "models/common/batch.h"
#include "core/kda_replay_records.h"
#include "models/glm5next/execution_shape.h"
#include "models/glm5next/state/context.h"

namespace ninfer::models::glm5next {

struct AttentionLayout {
    TensorRegion positions, valid_columns, table_rows;
};

struct BatchLayout {
    TensorRegion ids, positions, valid_columns;
    LayoutRegion ingress;
    AttentionLayout attention;
};

struct RowLayout {
    TensorRegion live_rows;
    TensorRegion source_offsets, destination_offsets, base_positions, accepted_columns,
        count_columns;
    TensorRegion anchors, count_slots, sampling_positions, sampling_columns;
    LayoutRegion ingress;
};

struct AttentionFrameLayout {
    TensorRegion to_global, from_gather, live_rows, live_columns, source_offsets,
        destination_offsets;
    TensorRegion main_tables, mtp_tables, base_positions, record_offsets;
    TensorRegion accepted_columns, active_columns;
    TensorRegion selection_rows;
    LayoutRegion ingress;
};

struct SpeculativeFrameLayout {
    TensorRegion drafts, extents, verify_ids, last_columns;
};

struct DFlashFrameLayout {
    TensorRegion features, ids, positions, valid_columns, proposal_rows, window_count;
    TensorRegion candidate_ids, proposal_q;
};

struct MtpFrameLayout {
    BatchLayout history;
    std::vector<BatchLayout> draft_steps;
    TensorRegion conditioning, previous_hidden, recursive_hidden;
    TensorRegion proposal_sampling;
    std::optional<TensorRegion> target_tokens;
    TensorRegion skip_first;
    ops::KpoolBlockSelectionLayout selection;
    TensorRegion tail_keys, tail_gates;
    std::array<TensorRegion, 3> tail_offsets;
};

struct RaggedLayout {
    TensorRegion cu_seqlens, seq_lengths, token_to_seq;
};

struct PackedMtpLayout {
    RaggedLayout ragged;
    TensorRegion input_segments, skip_first, base_positions, table_rows;
    TensorRegion source_offsets, destination_offsets;
};

struct PackedDFlashLayout {
    RaggedLayout ragged;
    TensorRegion source_indices, local_indices;
};

struct PackedPrefillLayout {
    RaggedLayout global, local;
    TensorRegion source_offsets, destination_offsets, local_tables, sampled_offsets;
    std::optional<PackedMtpLayout> mtp;
    std::optional<PackedDFlashLayout> dflash;
};

struct FrameLayout {
    BatchLayout main, decode;
    RowLayout rows;
    AttentionFrameLayout attention;
    LayoutRegion ingress, commit_ingress, host_ingress, sampling_ingress, egress, result_egress;
    TensorRegion prefill_hidden, pending_hidden;
    std::optional<KdaReplayRecordLayout> kda_records;
    ops::KpoolRawRecordLayout main_records;
    std::optional<SpeculativeFrameLayout> speculative;
    std::optional<MtpFrameLayout> mtp;
    std::optional<DFlashFrameLayout> dflash;
    std::optional<PackedPrefillLayout> packed_prefill;
    std::optional<TensorRegion> logits, token_counts, sampling, grammar_masks;
    std::optional<TensorRegion> candidates, gathered_candidates, sampled_ids, output_counts;
    std::optional<TensorRegion> top1_probabilities, draft_confidences;
    std::optional<TensorRegion> visual_indices;
};

void reshape_frame_controls(FrameLayout&, PhysicalBatchShape);

[[nodiscard]] FrameLayout plan_frame(LayoutBuilder&, const execution::ModelProfile&,
                                     const ExecutionShape&);

} // namespace ninfer::models::glm5next
