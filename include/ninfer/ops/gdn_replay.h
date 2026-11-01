#pragma once

#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"

#include "core/device.h"

#include <cstdint>
#include <span>

namespace ninfer::ops {

struct GdnReplayFoldRow {
    std::int64_t source_offset;
    std::int64_t destination_offset;
    std::int32_t commit_columns;
};

/**
 * Replays each row's [0,commit_columns) record prefix across all GDN layers. Row b addresses
 * physical record row b; rows are not filtered, compressed or reordered. Positive rows read
 * source_offset and publish to destination_offset, both arena-relative byte offsets selecting
 * complete StateImage records. Each selected field is dense, aligned and in bounds. A row may
 * update in place; active destinations are distinct and cannot overwrite another active source.
 *
 * commit_columns is in [0,T]. Zero is a strict no-op: no record or state is accessed. A positive
 * row writes final FP32 recurrent state and tail_3(old_history || conv_record[0:commit_columns]).
 * Both regions are bit-identical to the corresponding snapshot of the same full physical
 * record-producing block from the same initial state and represented inputs. Projection is
 * not re-evaluated at the commit width.
 *
 * Construction validates fixed state/record geometry. The bound arena, field geometry and
 * record addresses stay stable for the plan lifetime; record byte offsets may change. Host row
 * descriptors are captured by value. Device I64 offset tables and I32 commit extents are read
 * on replay. Metadata and records are preserved and must not overlap selected state fields.
 * Active rows are in [1,record_capacity]. No workspace or device allocation is owned by the Op.
 */
class GdnReplayFoldPlan {
public:
    GdnReplayFoldPlan(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states);

    void execute(std::span<const GdnReplayFoldRow> rows, CudaExecutionView execution) const;

    // Device I64 [B] offsets and I32 [B] commit lengths, 1<=B<=record_capacity, obey
    // the same row semantics as the Host overload, including strict zero-row no-op.
    // Metadata is contiguous, disjoint from record/state storage, and preserved. Replay
    // consumes its current values; the caller guarantees valid selectors and write separation.
    void execute(const Tensor& source_offsets, const Tensor& destination_offsets,
                 const Tensor& commit_columns, CudaExecutionView execution) const;

private:
    GdnReplayRecords records_;
    LinearAttentionStateAllLayersView states_;
};

} // namespace ninfer::ops
