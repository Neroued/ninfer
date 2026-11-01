#pragma once

// Two-device (tp 2) forms of linear_swiglu. Our file: ninfer/ops/linear_swiglu.h is upstream's and
// carries no hook.
//
// Shapes registered for the two-device forms: NVFP4 and FP8 register the output-row half,
// weight [17408,5120], x [5120,T], out [8704,T], with the same routes as the whole problem; NVFP4
// also the half of the dense FFN-14336 gate|up, weight [14336,5120], out [7168,T]. Gate rows [0,M)
// precede their matching up rows [M,2M), M = N/2.

#include "core/device.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/linear_swiglu.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * @brief Column-parallel (output-split) linear_swiglu across two devices.
 *
 * @details Rank r's shard `gate_up_weight[r]` holds its block of the gate rows followed by the
 * matching block of the up rows: for the [34816,5120] problem split evenly, gate rows
 * `[r*8704,(r+1)*8704)` then up rows `[17408+r*8704,17408+(r+1)*8704)`, a standalone
 * [17408,5120] tensor in the single-device gate/up layout. Rank r computes
 * `out[r] = linear_swiglu(x[r], gate_up_weight[r])` from the activation replicated on both ranks;
 * the two output blocks concatenate along `ne[0]` into the single-device result, and each block is
 * directly rank r's input block of the row-parallel down projection. Nothing is communicated, so
 * each rank's numerical contract is that of linear_swiglu() at the shard shape, which must be a
 * registered problem.
 *
 * Both ranks must agree on the weight format, on `K`, and on `T`. Every requirement of
 * linear_swiglu() applies per rank; `x[r]`, `gate_up_weight[r]`, `out[r]` and `workspace[r]` must
 * be resident on `ec.dev[r]`, and rank r's work is enqueued on `ec.dev[r]->stream`. The caller
 * obligation of ninfer/ops/allreduce.h applies: inputs staged on a device's legacy default stream
 * must be retired before the call. The call does not synchronize and preserves the caller's
 * current CUDA device.
 *
 * @param[in] x Per-rank BF16 activation `[K,T]`, identical on both ranks.
 * @param[in] gate_up_weight Per-rank gate/up shard `[N_r,K]`.
 * @param[out] out Per-rank BF16 output block `[N_r/2,T]`.
 * @param[in] policy Permitted private activation-compute profiles, applied to both ranks.
 * @param[in,out] workspace Per-rank caller-owned transient arena, sized by
 * linear_swiglu_workspace_capacity_bytes() at the shard shape. It may be null when that capacity
 * is zero.
 * @param[in] ec Execution context holding two distinct devices.
 */
void linear_swiglu_column_parallel(const std::array<Tensor, 2>& x,
                                   const std::array<Weight, 2>& gate_up_weight,
                                   const std::array<Tensor, 2>& out, LinearPolicy policy,
                                   const std::array<WorkspaceArena*, 2>& workspace,
                                   const ExecutionContext& ec);

/// A16-only column-parallel form; it passes no workspace, which the NVFP4 and FP8 A16 routes do
/// not need.
/// Model execution passes a policy; this form is the A16 entry the op qualification suites use.
void linear_swiglu_column_parallel(const std::array<Tensor, 2>& x,
                                   const std::array<Weight, 2>& gate_up_weight,
                                   const std::array<Tensor, 2>& out, const ExecutionContext& ec);

} // namespace ninfer::ops
