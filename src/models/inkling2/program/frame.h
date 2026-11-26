#pragma once

#include "models/common/batch.h"
#include "models/inkling2/execution/profile.h"
#include "models/inkling2/execution_shape.h"

#include <optional>

namespace ninfer::models::inkling2 {

struct BatchLayout {
    TensorRegion ids;
    TensorRegion positions;
};

struct RowLayout {
    TensorRegion live_rows;
    TensorRegion valid_columns;
    TensorRegion source_offsets;
    TensorRegion destination_offsets;
    TensorRegion main_tables;
    TensorRegion dspark_tables;
    TensorRegion base_positions, proposal_rows, proposal_columns, draft_extents, snapshot_slots;
    TensorRegion count_slots;
    TensorRegion sampling_positions;
    TensorRegion count_columns;
    TensorRegion anchors;
};

struct RaggedLayout {
    TensorRegion cu_seqlens;
    TensorRegion seq_lengths;
    TensorRegion token_to_seq;
};

struct PackedPrefillLayout {
    RaggedLayout ragged;
    TensorRegion source_offsets;
    TensorRegion destination_offsets;
    TensorRegion tables, dspark_tables;
    TensorRegion sampled_offsets;
};

struct VerifyLayerLayout {
    TensorRegion k_conv, v_conv, attn_conv, mlp_conv;
    TensorRegion k, v, k_scales, v_scales;
};

// Each recurrent step consumes an independent, 16-byte-aligned I32 token vector.
[[nodiscard]] constexpr int draft_batch_stride(int rows) { return (rows + 3) & ~3; }

struct DSparkFrameLayout {
    TensorRegion prefill_aux, pending_aux;
    RaggedLayout swa_ragged;
    TensorRegion swa_source_indices, last_aux_indices;
    TensorRegion draft_ids, draft_probabilities, target_ids;
    TensorRegion accepted_columns, last_columns;
    LayoutRegion commit_ingress;
    std::vector<VerifyLayerLayout> layers;
    // Indexed by (physical rows - 1) * max_drafts + (proposal columns - 1).
    std::vector<TensorRegion> stem_indices, tail_indices;
};

struct FrameLayout {
    std::optional<DSparkFrameLayout> dspark;
    TensorRegion top1_probabilities;
    LayoutRegion draft_egress;
    BatchLayout main;
    BatchLayout decode;
    RowLayout rows;
    LayoutRegion ingress;
    LayoutRegion host_ingress;
    LayoutRegion sampling_ingress;
    LayoutRegion egress;
    TensorRegion prefill_hidden;
    TensorRegion pending_hidden;
    PackedPrefillLayout packed;
    std::optional<TensorRegion> visual_indices;
    std::optional<TensorRegion> logits;
    std::optional<TensorRegion> token_counts;
    std::optional<TensorRegion> sampling;
    std::optional<TensorRegion> grammar_masks;
    std::optional<TensorRegion> sampled_ids;
    std::optional<TensorRegion> output_counts;
};

void reshape_frame_controls(FrameLayout&, PhysicalBatchShape);

[[nodiscard]] FrameLayout plan_frame(LayoutBuilder&, const execution::ModelProfile&,
                                     const ExecutionShape&);

} // namespace ninfer::models::inkling2
