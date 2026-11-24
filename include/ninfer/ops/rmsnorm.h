#pragma once

#include "core/tensor.h"
#include "core/device.h"


namespace ninfer::ops {

/**
 * Applies RMSNorm over the fastest dimension D=ne[0]. For each logical row r:
 *
 *   inv_r    = 1 / sqrt((1/D) * sum_d x[d,r]^2 + eps)
 *   gain[d]  = unit_offset ? 1 + weight[d] : weight[d]
 *   ideal[d,r] = x[d,r] * inv_r * gain[d].
 *
 * `x` and `out` are same-shaped contiguous BF16 tensors, weight is contiguous BF16 [D], and eps
 * is positive and finite. Nonnegative empty shapes perform no device work. Nonempty backing is
 * naturally BF16-aligned. Input, weight, and output must not overlap.
 * Preserves inputs and writes out; no workspace or persistent state.
 */
void rmsnorm(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, Tensor& out,
             CudaExecutionView execution);

/**
 * Applies RMS normalization over ne[0] and an elementwise SiLU gate. For each logical row r:
 *
 *   inv_r    = 1 / sqrt((1/D) * sum_d x[d,r]^2 + eps)
 *   ideal[d,r] = x[d,r] * inv_r * weight[d] * SiLU(z[d,r]).
 *
 * `x`, `z`, and `out` are same-shaped contiguous BF16 tensors with positive extents,
 * `weight` is contiguous BF16 [D], and eps is positive and finite. This form does not apply a
 * unit offset to weight. The output must not overlap any input; read-only inputs may share
 * backing. Inputs remain unchanged and the output is fully written, without a required BF16
 * norm intermediate. No workspace or persistent state. execution supplies the stream and
 * the device's positive physical SM count.
 */
void silu_gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& z, float eps,
                        Tensor& out, CudaExecutionView execution);

/**
 * Applies RMS normalization over ne[0] and an elementwise sigmoid gate. For each logical row r:
 *
 *   inv_r        = 1 / sqrt((1/D) * sum_d x[d,r]^2 + eps)
 *   ideal[d,r]   = x[d,r] * inv_r * weight[d] * sigmoid(z[d,r]).
 *
 * Shapes, direct weight multiplier, effects, non-overlap, zero-workspace and execution rules
 * are the same as silu_gated_rmsnorm. Every positive D and remaining extent is supported
 * within the view's represented range and available storage. Norm and gate form one computation,
 * without a required BF16 norm intermediate.
 */
void sigmoid_gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& z, float eps,
                           Tensor& out, CudaExecutionView execution);

} // namespace ninfer::ops
