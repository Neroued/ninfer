#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Matrix projection, independently for every token column:
 *
 *   out[n,t] = BF16(sum_{k=0..K-1} dequantize(w)[n,k] * float(x[k,t])).
 *
 * `x` is contiguous BF16 [K,T], `w` has logical shape [N,K], and `out` is contiguous BF16
 * [N,T], with the first index stored fastest. Supported weight encodings are contiguous
 * BF16_CTRL and FP32_CTRL, plus RowSplit Q4G64_F16S, Q5G64_F16S, Q6G64_F16S, and W8G32_F16S
 * with FP16 scales. The selected backend determines FP32 accumulation grouping but preserves the
 * logical dequantized matrix product and its BF16 output boundary. Every RowSplit format owns a
 * finite registry of exact physical problems and selects its kernel internally; the encoding and
 * alignment contract alone do not imply arbitrary-shape support. `out` must not overlap x or any
 * weight plane. `ws` supplies transient policy scratch and carries no semantic state beyond the
 * call.
 */
void linear(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops
