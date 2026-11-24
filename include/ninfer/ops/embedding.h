#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"

namespace ninfer::ops {

/**
 * out[d,t] = decode(table)[ids[t],d].
 *
 * Table represents [vocab,D], ids is contiguous I32 [T] with values in [0,vocab), and out is
 * contiguous BF16 [D,T]. Supports contiguous BF16 and registered RowSplit Q6-G64/Q8-G32
 * geometries with FP16 group scales, and RowScale E4M3FN with one BF16 multiplier per row.
 * FP8 output must be four-byte aligned. Dense BF16 gathers are bit-exact; quantized gathers
 * decode and scale the stored values before the final BF16 store.
 *
 * Writes all of out, which must not overlap ids or any table plane. Inputs are preserved;
 * an empty token extent does no work. No workspace or other state is used.
 */
void embedding(const Tensor& ids, const Weight& table, Tensor& out, CudaExecutionView execution);

/**
 * x[d,p] += sum_{c=0..3} table[d,indices[c,p]] * weights[c,p].
 *
 * Table [D,R] and x [D,P] are contiguous BF16, indices [4,P] is contiguous I32 with values
 * in [0,R), and weights [4,P] is contiguous FP32. Inputs do not overlap x. Computes the
 * four-term weighted gather and residual addition, then stores BF16 in place; no intermediate
 * BF16 value is prescribed. Inputs are preserved. An empty x does no work. No workspace or
 * other state is used.
 */
void embedding_interpolate_add(const Tensor& table, const Tensor& indices, const Tensor& weights,
                               Tensor& x, CudaExecutionView execution);

} // namespace ninfer::ops
