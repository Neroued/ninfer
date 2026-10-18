#pragma once

#include "core/device.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/projection_weights.h"


#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

struct GdnConvOutputs {
    Tensor query, key, value, z;
};

struct GdnConvSourceState {
    Tensor history;      // Read-only BF16 [C,3,S].
    Tensor source_slots; // Read-only device I32 [B], absolute slots in [0,S).
};

struct GdnConvSnapshotState {
    Tensor history;        // BF16 [C,3,S], source and destination pool.
    Tensor source_slots;   // Read-only device I32 [B], absolute slots in [0,S).
    Tensor snapshot_bases; // Read-only device I32 [B], reserves [base,base+T).
};

/**
 * Op: gdn_input_proj_conv_snapshot
 *
 * Math / indexing:
 *   Let p=concat(Wq*x,Wk*x,Wv*x) and z=Wz*x. For each batch row b, read the width-three
 *   history selected by source_slots[b]. At each valid token t, evaluate the four-tap
 *   causal depthwise convolution over history and p[:,t,b], apply SiLU, and split its
 *   channels to query/key/value. Publish the resulting width-three history at
 *   snapshot_bases[b]+t. Z bypasses convolution.
 *
 * Logical shapes / supported domain:
 *   H=input rows, Q=query/key rows (2048 unless stated), V=value/z rows, C=2*Q+V.
 *   Single Q8-G32/FP16 weights use [12288,2048] (V=4096), [16384,2560] (V=6144), or
 *   on DGX Spark [8192,2560] (Q=1024,V=3072). FP8 row-scale and NVFP4 block-scale
 *   weights use [16384,5120] (V=6144). Paired Q4-G64/FP16 [4096,5120] and Q5-G64/FP16
 *   [12288,5120] use V=6144. The logical row order is [q,k,v,z]. Q8 and paired weights
 *   use A16. Single-parent LinearPolicy permits A8 for FP8 or A4 for NVFP4.
 *
 *   x BF16 [H,T,B], conv_weight BF16 [C,4], history BF16 [C,3,S], query/key BF16 [Q,T,B],
 *   value/z BF16 [V,T,B]. History has dense [C,3] slots with explicit nb[2]; other tensors
 *   are contiguous. Every slot starts 16-byte aligned. Selectors are contiguous device I32 [B].
 *   B=1 accepts every positive T; B=2..8 accepts T=1..16. Empty valid_columns means every
 *   row is dense; otherwise it is device I32 [B] with each extent in [1,T]. Projection
 *   processes all M=B*T safe columns.
 *
 * Numeric / effects:
 *   Projection, convolution and SiLU form one computation. Private projected values are
 *   not an observable BF16 boundary. Outputs and published histories are BF16.
 *   Query/key/value are exact zero in invalid tails; z is written for every physical
 *   column. Only valid destination history slots are written. The caller reserves disjoint
 *   complete [base,base+T) intervals, prevents a row from overwriting another row's source,
 *   and may overlap its own source with its destination after the initial history is loaded.
 *   Other slots are unchanged. Tensor operands are disjoint except that the two read-only
 *   state selector tensors may alias. Weight parents and call-local scratch overlap none
 *   of the tensors. Device selector values and cross-row slot relationships are caller promises.
 *
 * Workspace / execution:
 *   Uses scoped storage at the caller arena cursor; earlier allocations may hold operands.
 *   A zero capacity performs no allocation. The query covers the complete computation.
 * Capture/replay uses execution.stream() and consumes device metadata at replay.
 */
void gdn_input_proj_conv_snapshot(const Tensor& x, const ProjectionWeights& weights,
                                  const Tensor& conv_weight, GdnConvSnapshotState state,
                                  const Tensor& valid_columns, GdnConvOutputs outputs,
                                  MemoryArena& workspace, CudaExecutionView execution);

[[nodiscard]] std::size_t
gdn_input_proj_conv_snapshot_workspace_capacity_bytes(const ProjectionProfile& weights,
                                                      std::int32_t batch, std::int32_t min_tokens,
                                                      std::int32_t max_tokens);

/**
 * Op: gdn_input_proj_conv_record
 *
 * Evaluates the same projection/convolution/SiLU computation from a read-only source history.
 * Supported weights, tensor layouts, valid prefixes and output effects match snapshot above;
 * the record domain is B=1..8, T=2..16. conv_record is contiguous, 16-byte-aligned BF16 [C,T,B].
 * Only its valid prefix is defined. It receives each newest projected history column.
 *
 * Query/key/value/z and valid record columns are bit-identical to snapshot from the same
 * initial history, represented inputs, physical T/B and activation permission. Records equal
 * the newest column of each corresponding snapshot. Source history is never modified.
 * The record's BF16 publication does not require rounding private convolution intermediates.
 *
 * All tensor operands are disjoint from one another, the weight parents and call-local scratch.
 * Materialized computation may write projection directly to conv_record. Workspace queries
 * cover the complete Op and the supplied inclusive T interval. Capture/replay is supported.
 */
void gdn_input_proj_conv_record(const Tensor& x, const ProjectionWeights& weights,
                                const Tensor& conv_weight, GdnConvSourceState state,
                                const Tensor& valid_columns, GdnConvOutputs outputs,
                                Tensor& conv_record, MemoryArena& workspace,
                                CudaExecutionView execution);

[[nodiscard]] std::size_t
gdn_input_proj_conv_record_workspace_capacity_bytes(const ProjectionProfile& weights,
                                                    std::int32_t batch, std::int32_t min_tokens,
                                                    std::int32_t max_tokens);

[[nodiscard]] inline std::size_t gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
    const ProjectionWeights& weights, std::int32_t batch, std::int32_t first, std::int32_t last) {
    return gdn_input_proj_conv_snapshot_workspace_capacity_bytes(projection_profile(weights), batch,
                                                                 first, last);
}

[[nodiscard]] inline std::size_t gdn_input_proj_conv_record_workspace_capacity_bytes(
    const ProjectionWeights& weights, std::int32_t batch, std::int32_t first, std::int32_t last) {
    return gdn_input_proj_conv_record_workspace_capacity_bytes(projection_profile(weights), batch,
                                                               first, last);
}
} // namespace ninfer::ops
