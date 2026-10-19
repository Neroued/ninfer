#pragma once

#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"

#include "core/device.h"

#include <cstdint>
#include <span>

namespace ninfer::ops {

struct GdnReplayFoldRow {
    std::int32_t source_state_slot;
    std::int32_t destination_state_slot;
    std::int32_t commit_columns;
};

/**
 * Prepared Op: gdn_replay_fold
 *
 * Replays each row's [0,commit_columns) record prefix across every registered GDN layer and
 * updates the caller-selected absolute linear-attention destination state slot. rows[b] always maps
 * to physical record row b; rows are not filtered, compressed, or reordered. The active row count
 * is rows.size() and must be in [1,records.spec.record_capacity].
 *
 * Each layer has dense inner state; its Tensor carries the slot stride and the view carries
 * independent layer strides. Distinct fields/layers must not overlap their actual data blocks.
 * source_state_slot and destination_state_slot are in [0,states.conv_layer0.ne[2]). A row may be
 * in-place. Destinations are distinct and cannot overwrite another active row's source.
 * source_state_slot is the absolute slot used to produce that row's records. commit_columns is in
 * [0,T]. Zero is a
 * strict no-op for the row: no record or state is read and neither recurrent state nor convolution
 * history is written. For a positive extent, the Op consumes raw key/value/{g,beta} records in
 * order, writes the final FP32 recurrent state, and sets convolution history to
 * tail_3(old_history || conv_record[0:commit_columns]). Both state regions must be bit-identical
 * to the corresponding Nth snapshot of the same full physical record-producing block, with the
 * same initial state and represented inputs. Projection is not re-evaluated at the commit width.
 *
 * Construction validates the immutable record/state geometry, layout, address ranges and
 * disjointness once. execute() validates only active rows, state selectors and commit extents.
 * The bound record/state addresses must remain stable for the plan lifetime. CUDA Graph capture
 * copies row descriptors by value; replay consumes record/state contents at the bound addresses,
 * while changing host row descriptors requires a new capture.
 *
 * Registered (layers, Q/K heads, V heads, conv channels) geometries are (48,16,48,10240),
 * (30,16,32,8192), (36,16,48,10240), and (36,8,24,5120). The Op owns no workspace or device
 * allocation, and does not read query or generate token output. The four record planes are
 * read-only, disjoint, and do not overlap either state region.
 */
class GdnReplayFoldPlan {
public:
    GdnReplayFoldPlan(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states);

    void execute(std::span<const GdnReplayFoldRow> rows, CudaExecutionView execution) const;

    // Device I32 [B] metadata, 1<=B<=record_capacity. Selectors and commit lengths obey
    // the same row semantics as the Host overload, including strict zero-row no-op.
    // Metadata is contiguous, disjoint from record/state storage, and preserved. Replay
    // consumes its current values; the caller guarantees valid selectors and write separation.
    void execute(const Tensor& source_slots, const Tensor& destination_slots,
                 const Tensor& commit_columns, CudaExecutionView execution) const;

private:
    GdnReplayRecords records_;
    LinearAttentionStateAllLayersView states_;
};

} // namespace ninfer::ops
