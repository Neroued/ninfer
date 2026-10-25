#pragma once

#include "core/kda_replay_records.h"
#include "models/glm5next/state/context.h"

namespace ninfer::models::glm5next {

struct ExecutionShape {
    std::uint32_t max_context, max_concurrency, prefill_chunk, draft_tokens;
    KvCacheStorage kv_storage;
    EnginePurpose purpose;
    bool context_cache, gpu_timing;
    bool operator==(const ExecutionShape&) const = default;
};

struct BatchLayout {
    TensorRegion ids, positions, valid_columns, source_offsets, destination_offsets, table_rows;
    TensorRegion base_positions, record_offsets, accepted_columns;
    TensorRegion local_to_global, global_to_local, selection_rows;
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
    BatchLayout main, owner;
    LayoutRegion ingress, sampling_ingress, egress;
    TensorRegion prefill_hidden, pending_hidden;
    std::optional<KdaReplayRecordLayout> kda_records;
    ops::KpoolRawRecordLayout main_records;
    std::optional<MtpFrameLayout> mtp;
    std::optional<TensorRegion> logits, token_counts, sampling, grammar_masks;
    std::optional<TensorRegion> candidates, gathered_candidates, sampled_ids, output_counts;
    std::optional<TensorRegion> visual_indices;
    TensorRegion branch_exchange;
};

[[nodiscard]] FrameLayout plan_frame(LayoutBuilder&, const execution::ModelProfile&,
                                     const ExecutionShape&);

} // namespace ninfer::models::glm5next
