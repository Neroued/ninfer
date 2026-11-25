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
    TensorRegion live_rows, valid_columns, source_offsets, destination_offsets, main_tables;
    TensorRegion base_positions, record_offsets, accepted_columns, count_columns;
    TensorRegion raw_tail, raw_tail_offsets, count_slots, sampling_positions;
    std::uint32_t max_positions;
};

struct MtpBatchLayout {
    TensorRegion ids, positions, rope_positions;
    TensorRegion live_rows, valid_columns, source_offsets, destination_offsets, table_rows;
    TensorRegion base_positions, record_offsets;
    std::optional<TensorRegion> visual_indices;
    LayoutRegion ingress;
};

struct MtpBufferLayout {
    MtpBatchLayout history;
    std::vector<MtpBatchLayout> draft_steps;
    TensorRegion base_conditioning, alignment_conditioning, hidden, sample;
    TensorRegion tail_keys, tail_coordinates, tail_key_offsets, tail_coordinate_offsets;
    TensorRegion drafts, extents, verify_ids, target_tokens;
    TensorRegion last_columns, active_columns, skip_first, previous_rope, history_base_positions;
};

struct PackedInputLayout {
    TensorRegion ids, positions, rope_positions;
    TensorRegion cu_seqlens, seq_lengths, token_to_seq;
    TensorRegion source_offsets, destination_offsets, table_rows;
    std::optional<TensorRegion> raw_tail, raw_tail_offsets;
    LayoutRegion ingress;
};

struct BufferLayout {
    BatchLayout batch, decode;
    TensorRegion anchors;
    LayoutRegion ingress, commit_ingress, host_region;
    TensorRegion commit_destination_offsets;
    TensorRegion prefill_sample;
    std::optional<TensorRegion> prefill_conditioning;
    std::optional<PackedInputLayout> packed_main, packed_history;
    std::optional<TensorRegion> prefill_sampled_offsets, prefill_sampled_hidden, prefill_aligned;
    TensorRegion pending_sample;
    std::optional<TensorRegion> pending_conditioning;
    QsaRecordLayout main_qsa;
    TensorRegion ple_records;
    std::optional<GdnReplayRecordLayout> gdn_records;
    std::optional<MtpBufferLayout> mtp;
    std::optional<TensorRegion> logits;
    std::optional<TensorRegion> ple_embedding, vocabulary_local, vocabulary_gathered;
    std::optional<TensorRegion> proposal_sampling;
    std::optional<TensorRegion> sampled_ids, output_counts;
    std::optional<TensorRegion> top1_probabilities, draft_confidences;
    std::optional<TensorRegion> token_counts, sampling, grammar_masks;
};

[[nodiscard]] MtpBatchLayout history_control_view(const MtpBatchLayout&, std::uint32_t tokens,
                                                  std::uint32_t rows);
[[nodiscard]] PackedInputLayout packed_input_view(const PackedInputLayout&, std::int32_t tokens,
                                                  std::int32_t segments);

void select_buffer_rows(BufferLayout&, const BufferLayout& backing, std::uint32_t rows);

[[nodiscard]] BufferLayout plan_buffers(LayoutBuilder&, const Config&, const EngineOptions&,
                                        std::uint32_t chunk, std::uint32_t public_tokens,
                                        const GdnConfig&, std::uint32_t local_vocabulary,
                                        std::uint32_t ranks, bool device_ple);
} // namespace ninfer::models::qwen4_exp
