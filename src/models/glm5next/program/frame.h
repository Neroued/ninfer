#pragma once

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

struct MtpFrameLayout {
    BatchLayout history;
    std::vector<BatchLayout> draft_steps;
    TensorRegion conditioning, previous_hidden, recursive_hidden, drafts, extents, verify_ids;
    TensorRegion proposal_sampling, proposal_columns;
    std::optional<TensorRegion> target_tokens;
    TensorRegion skip_first, last_columns;
    ops::KpoolBlockSelectionLayout selection;
    TensorRegion tail_keys, tail_gates;
    std::array<TensorRegion, 3> tail_offsets;
};

struct FrameLayout {
    BatchLayout main, decode;
    RowLayout rows;
    AttentionFrameLayout attention;
    LayoutRegion ingress, commit_ingress, host_ingress, sampling_ingress, egress, result_egress;
    TensorRegion prefill_hidden, pending_hidden;
    std::optional<KdaReplayRecordLayout> kda_records;
    ops::KpoolRawRecordLayout main_records;
    std::optional<MtpFrameLayout> mtp;
    std::optional<TensorRegion> logits, token_counts, sampling, grammar_masks;
    std::optional<TensorRegion> candidates, gathered_candidates, sampled_ids, output_counts;
    std::optional<TensorRegion> visual_indices;
};

[[nodiscard]] FrameLayout plan_frame(LayoutBuilder&, const execution::ModelProfile&,
                                     const ExecutionShape&);

} // namespace ninfer::models::glm5next
