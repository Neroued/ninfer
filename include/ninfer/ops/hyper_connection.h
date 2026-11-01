#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Op: grouped offset RMSNorm
 *
 * Math / indexing:
 *   inv[g,t]   = 1 / sqrt((1/H) * sum_h x[g*H+h,t]^2 + eps)
 *   out[i,t]   = x[i,t] * inv[floor(i/H),t] * (1 + weight[i])
 *
 * Logical shapes:
 *   x, out [G*H, T]; weight [G*H]. Every column t and group g normalizes its own H-wide slice.
 *
 * Supported domain:
 *   BF16, contiguous, G >= 1 dividing the row width, T >= 1, eps > 0 and finite.
 *
 * Numeric:
 *   FP32 reduction; one BF16 rounding of each output. The oracle is FP64 over represented inputs.
 *
 * Effects:
 *   Writes `out`; `x` and `out` must not overlap.
 *
 * Workspace: none.  Execution: stream ordered, CUDA Graph capturable.
 */
void grouped_offset_rmsnorm(const Tensor& x, const Tensor& weight, std::int32_t groups, float eps,
                            Tensor& out, cudaStream_t stream);

/**
 * Op: hyper-connection residual expansion
 *
 * Math / indexing:
 *   residual[s*H+h,t] = x[h,t] for every stream s.
 *
 * Logical shapes: x [H,T], residual [S*H,T], S = `streams` >= 1.
 * Supported domain: BF16, contiguous.  Numeric: exact copy.
 * Effects: writes `residual`; no overlap with `x`.  Workspace: none.
 * Execution: stream ordered, CUDA Graph capturable.
 */
void hyper_connection_expand(const Tensor& x, std::int32_t streams, Tensor& residual,
                             cudaStream_t stream);

/**
 * Op: hyper-connection combine (inject one block output into every residual stream)
 *
 * Math / indexing:
 *   residual[s*H+h,t] = residual[s*H+h,t] + inject[s,t] * y[h,t]
 *
 * Logical shapes: y [H,T], inject [S,T], residual [S*H,T].
 * Supported domain: y, residual BF16; inject FP32 (the `inject` output of hyper_connection_mix);
 *   contiguous.
 * Numeric: FP32 product and sum, one BF16 rounding of each updated residual element.
 * Effects: updates `residual` in place; `y` and `inject` must not overlap it.
 * Workspace: none.  Execution: stream ordered, CUDA Graph capturable.
 */
void hyper_connection_combine(const Tensor& y, const Tensor& inject, Tensor& residual,
                              cudaStream_t stream);

/**
 * Weights of one hyper-connection mixer: S = 4 streams of width H, low rank R (the geometries each
 * format serves are listed under hyper_connection_mix).
 *
 *   norm: BF16 [S*H] grouped offset-RMSNorm weight.
 *   down: Q8_G32_FP16 row-split [R + I, S*H]. Rows [0,R) are the low-rank down projection; when
 *         I = S, rows [R, R+S) are the block-inject projection (the converter groups both into one
 *         parent). The output mixer of the model has no inject rows (I = 0).
 *   up:   Q8_G32_FP16 row-split [S*H, R].
 *
 * Options: both projections BF16 (down [R + I, S*H], up [S*H, R], as a checkpoint stores them);
 * or, not in a recipe and of unqualified quality, both NVFP4 (BlockScaleK16M128x4, E4M3 group
 * scales, FP32 divisor), down stored as [round_up(R + I, 128), S*H] with the rows past R + I
 * unused, up [S*H, R], supported for T <= 8, H = 2560 and R = 320 only.
 */
struct HyperConnectionWeights {
    Tensor norm;
    Weight down;
    Weight up;
};

/**
 * Op: hyper-connection mix (input mix of one block, optional inject gates)
 *
 * Math / indexing (per column t; s indexes streams, `/S` is division by the stream count):
 *   n          = grouped_offset_rmsnorm(residual, norm, groups = S, eps)        [S*H]
 *   p          = down . n                                                       [R + I]
 *   low[r]     = silu(p[r] / S)                                                 [R]
 *   m[i]       = sigmoid((up . low)[i])                                         [S*H]
 *   mixed[h]   = (1/S) * sum_s m[s*H+h] * n[s*H+h]                              [H]
 *   inject[s]  = 2 * sigmoid(p[R+s] / S)               (only when inject rows exist)
 * where `down` and `up` are their decoded Q8 values (code * group scale).
 *
 * Logical shapes:
 *   residual [S*H, T]; mixed [H, T]; inject [S, T]; T >= 1 (no upper bound).
 *
 * Supported domain:
 *   S = 4; eps > 0 and finite. residual, norm, mixed BF16; inject FP32; weights as
 *   HyperConnectionWeights. `inject` must be non-null exactly when `down` has R + S rows. By weight
 *   format: Q8 H in {2560, 5120} and R in {320, 640} (the Q8 Linear shapes registered for the
 *   general route); BF16 H and R positive multiples of 8; NVFP4 H = 2560, R = 320, T <= 8. A call
 *   outside the domain throws std::invalid_argument before anything is enqueued.
 *
 * Numeric:
 *   FP32 accumulation of both projections over exactly decoded weights. Implementations may stage
 *   the normalized activation and the low-rank activation in BF16 or FP16 (the activation operands
 *   of the contractions) and the projection p in BF16 (a Linear output). One BF16 rounding of
 *   `mixed`; `inject` is FP32. The oracle evaluates the complete formula in FP64 from represented
 *   inputs and decoded weights.
 *
 * Effects:
 *   Writes `mixed` and `inject`; `residual` is read only. Outputs must not overlap inputs.
 *
 * Workspace:
 *   hyper_connection_mix_workspace_bytes(min_tokens, max_tokens, inject, H, R).
 *
 * Execution:
 *   Stream ordered, no host synchronization, CUDA Graph capturable for every T. For T <= 8 and
 *   R = 320 on a device with thread block clusters (sm_90+), at H = 2560 (any weight format) or at
 *   H = 5120 with Q8 weights, the Op is two launches and needs no particular SM count or
 *   co-residency: a Down launch of thread block clusters, then an Up launch that is its
 *   programmatic dependent. Other geometries of the domain take the general route at every T. The
 *   Down launch is itself a programmatic dependent of the preceding kernel in the stream: it
 *   requests `norm` and the weights before that kernel completes, so these must not be written by
 *   it. The Op's tensor maps are encoded on the host at each call.
 */
void hyper_connection_mix(const Tensor& residual, const HyperConnectionWeights& weights, float eps,
                          Tensor& mixed, Tensor* inject, WorkspaceArena& workspace,
                          cudaStream_t stream);

/**
 * Op: hyper-connection combine followed by mix (one pending block output committed, then the next
 * mixer evaluated on the updated residual)
 *
 * Math / indexing:
 *   residual'  = hyper_connection_combine(previous_y, previous_inject, residual)   (BF16 values)
 *   mixed, inject = hyper_connection_mix(residual', weights)
 *
 * The mix reads the represented BF16 values of residual'; the result is the same as calling
 * hyper_connection_combine and then hyper_connection_mix.
 *
 * Logical shapes / domain: as the two composed Ops. previous_y [H,T] BF16, previous_inject [S,T]
 *   FP32, sharing no byte with `inject`: the small-T route reads previous_inject in the same
 *   launches that write inject, so overlapping buffers race.
 * Effects: updates `residual` in place to residual'; writes `mixed` and `inject`. A call outside
 *   the domain throws before anything is enqueued, leaving `residual` unchanged.
 * Workspace: hyper_connection_mix_workspace_bytes(min_tokens, max_tokens, inject, H, R).
 * Execution: as hyper_connection_mix.
 */
void hyper_connection_combine_mix(Tensor& residual, const Tensor& previous_y,
                                  const Tensor& previous_inject,
                                  const HyperConnectionWeights& weights, float eps, Tensor& mixed,
                                  Tensor* inject, WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Minimum caller workspace (bytes, from a 256-byte aligned cursor) sufficient for every
 * hyper_connection_mix / hyper_connection_combine_mix call with T in [min_tokens, max_tokens],
 * the given inject form, hidden width H = `hidden` and low rank R = `rank`. Throws for an empty or
 * non-positive interval, or for a hidden width or rank that is not a positive multiple of 8.
 */
[[nodiscard]] std::size_t hyper_connection_mix_workspace_bytes(std::int32_t min_tokens,
                                                               std::int32_t max_tokens, bool inject,
                                                               std::int32_t hidden,
                                                               std::int32_t rank);

} // namespace ninfer::ops
