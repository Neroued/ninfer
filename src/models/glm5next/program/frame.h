#pragma once

#include "core/kda_replay_records.h"
#include "models/glm5next/execution_shape.h"
#include "models/glm5next/state/context.h"

namespace ninfer::models::glm5next {

struct BatchLayout {
    TensorRegion ids, positions, valid_columns, source_offsets, destination_offsets, table_rows;
    TensorRegion base_positions, record_offsets, accepted_columns;
    TensorRegion count_slots, sampling_positions, selection_rows;
    LayoutRegion metadata;
};

struct MtpFrameLayout {
    BatchLayout history;
    std::vector<BatchLayout> draft_steps;
    TensorRegion conditioning, previous_hidden, recursive_hidden, drafts, extents, anchors, lengths;
    TensorRegion target_tokens, token_counts, proposal_sampling, proposal_columns;
    TensorRegion skip_first, last_columns, active_columns, accepted_drafts;
    ops::KpoolBlockSelectionLayout selection;
    TensorRegion tail_keys, tail_gates;
    std::array<TensorRegion, 3> tail_offsets;
};

struct FrameLayout {
    BatchLayout main;
    LayoutRegion ingress, sampling_ingress, egress, result_egress;
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
