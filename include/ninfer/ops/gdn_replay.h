#pragma once

#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"

#include <cuda_runtime.h>

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
 * source_state_slot and destination_state_slot are in [0,states.spec.slot_count). A row may be
 * in-place. Destinations are distinct and cannot overwrite another active row's source.
 * source_state_slot is the absolute slot used to produce that row's records. commit_columns is in
 * [0,T]. Zero is a
 * strict no-op for the row: no record or state is read and neither recurrent state nor convolution
 * history is written. For a positive extent, the Op consumes raw key/value/{g,beta} records in
 * order, writes the final FP32 recurrent state, and sets convolution history to
 * tail_3(old_history || conv_record[0:commit_columns]).
 *
 * Construction validates the immutable record/state geometry, layout, address ranges and
 * disjointness once. execute() validates only active rows, state selectors and commit extents.
 * The bound record/state addresses must remain stable for the plan lifetime.
 *
 * The Op admits the two registered all-layer geometries only, owns no workspace or device
 * allocation, and does not read query or generate token output. The four record planes are
 * read-only, disjoint, and do not overlap either state region.
 */
class GdnReplayFoldPlan {
public:
    GdnReplayFoldPlan(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states);

    void execute(std::span<const GdnReplayFoldRow> rows, cudaStream_t stream) const;

private:
    GdnReplayRecords records_;
    LinearAttentionStateAllLayersView states_;
};

} // namespace ninfer::ops
