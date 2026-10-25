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
    TensorRegion conditioning, recursive_hidden, drafts, extents, anchors;
    TensorRegion target_tokens, token_counts, embedding_enabled;
    ops::KpoolBlockSelectionLayout selection;
    StateImageLayout tail_image;
    ops::KpoolTailFields tail_fields;
    LayoutRegion tails;
    std::array<TensorRegion, 3> tail_offsets;
};

struct FrameLayout {
    BatchLayout main;
    LayoutRegion ingress, sampling_ingress, egress;
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
