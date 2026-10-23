#pragma once

#include "core/gdn_replay_records.h"
#include "models/qwen4_exp/config.h"

namespace ninfer::models::qwen4_exp {
struct QsaRecordLayout {
    TensorRegion keys, coordinates;
    std::uint32_t layers, width, batch;
};

struct BatchLayout {
    TensorRegion ids, positions, rope_positions;
    std::optional<TensorRegion> visual_indices;
    TensorRegion valid_columns, source_offsets, destination_offsets, main_tables;
    TensorRegion base_positions, record_offsets, accepted_columns;
    TensorRegion raw_tail, raw_tail_offsets, row_indices, count_slots, sampling_positions;
    std::uint32_t max_positions;
};

struct MtpBatchLayout {
    TensorRegion ids, positions, rope_positions;
    TensorRegion valid_columns, source_offsets, destination_offsets, table_rows;
    TensorRegion base_positions, record_offsets;
    std::optional<TensorRegion> visual_indices;
    LayoutRegion ingress;
};

struct MtpBufferLayout {
    MtpBatchLayout history;
    std::vector<MtpBatchLayout> draft_steps;
    LayoutRegion ingress;
    TensorRegion base_conditioning, alignment_conditioning, hidden, sample;
    TensorRegion tail_keys, tail_coordinates, tail_key_offsets, tail_coordinate_offsets,
        row_indices;
    TensorRegion drafts, extents, anchors, frontiers, target_tokens, token_counts;
    TensorRegion last_columns, active_columns;
};

struct BufferLayout {
    BatchLayout batch;
    TensorRegion prefill_sample;
    std::optional<TensorRegion> prefill_conditioning;
    TensorRegion pending_sample;
    std::optional<TensorRegion> pending_conditioning;
    QsaRecordLayout main_qsa;
    TensorRegion ple_records;
    std::optional<GdnReplayRecordLayout> gdn_records;
    std::optional<MtpBufferLayout> mtp;
    std::optional<TensorRegion> logits;
    std::optional<TensorRegion> ple_embedding, vocabulary_local, vocabulary_gathered;
    std::optional<TensorRegion> proposal_sampling;
    std::optional<TensorRegion> vision_features;
    std::optional<TensorRegion> sampled_ids, output_counts, accepted_drafts;
    std::optional<TensorRegion> token_counts, sampling, grammar_masks;
};

[[nodiscard]] BufferLayout plan_buffers(LayoutBuilder&, const Config&, const EngineOptions&,
                                        std::uint32_t chunk, std::uint32_t public_tokens,
                                        const GdnConfig&, std::uint32_t local_vocabulary,
                                        std::uint32_t ranks, bool device_ple);
} // namespace ninfer::models::qwen4_exp
