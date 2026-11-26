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
    TensorRegion tables;
    TensorRegion sampled_offsets;
};

struct FrameLayout {
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
