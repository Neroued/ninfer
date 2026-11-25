#pragma once

#include "core/kda_replay_records.h"
#include "core/linear_attention_state.h"

#include <span>

namespace ninfer::ops {

struct KdaReplayFoldRow {
    std::int64_t source_offset;
    std::int64_t destination_offset;
    std::int32_t commit_columns;
};

/**
 * For each positive commit m, fold records [0,m) across all layers. The source recurrent
 * field must contain the same initial S0 observed by kimi_delta_attention_replay_record
 * when it produced those records. The destination recurrent field (FP32 or BF16, matching the
 * planned state dtype) approximates the mathematical state Sm. Its convolution history is exactly
 * tail_3(source.conv || records.conv[0:m]) at the BF16 word level.
 * Record row b is fixed throughout the call; width>=1, 1<=B<=record_capacity, 0<=m<=width.
 *
 * State fields are dense and 256-byte aligned, selected by arena-relative I64 byte offsets.
 * Positive rows may update in place; destinations are distinct and cannot overwrite another
 * positive row's source. Zero rows access no state or records and may use -1 offsets.
 * All other fields, records and metadata are preserved. Caller protects every live source.
 * For Device rows the caller guarantees offset bounds, alignment and disjointness.
 *
 * The plan borrows stable arena/field geometry and frame records, never a permanent State payload
 * address. Host descriptors are captured by value; Device I64 offsets and I32 lengths are read
 * on replay. No workspace or device allocation is owned by this Op.
 */
class KdaReplayFoldPlan {
public:
    KdaReplayFoldPlan(const KdaReplayRecords& records, LinearAttentionStateAllLayersView states);
    void execute(std::span<const KdaReplayFoldRow> rows, CudaExecutionView execution) const;
    void execute(const Tensor& source_offsets, const Tensor& destination_offsets,
                 const Tensor& commit_columns, CudaExecutionView execution) const;

private:
    KdaReplayRecords records_;
    LinearAttentionStateAllLayersView states_;
};
} // namespace ninfer::ops
