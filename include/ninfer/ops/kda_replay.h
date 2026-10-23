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
 * Replays records [0,m) from source into destination across all layers, preserving the FP32
 * recurrent state of ordinary KDA updates on those represented inputs, bit for bit. Convolution
 * history becomes tail_3(source.conv || records.conv[0:m]). No projection or readout is performed.
 * Record row b is fixed throughout the call; 1<=B<=record_capacity<=8, 0<=m<=width.
 *
 * State fields are dense and 256-byte aligned, selected by arena-relative I64 byte offsets.
 * Positive rows may update in place; destinations are distinct and cannot overwrite another
 * positive row's source. Zero rows access no state or records and may use -1 offsets.
 * All other fields, records and metadata are preserved. Caller protects every live source.
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
