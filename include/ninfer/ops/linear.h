#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Matrix projection, independently for every token column:
 *
 *   out[n,t] = BF16(sum_{k=0..K-1} dequantize(w)[n,k] * x[k,t]).
 *
 * `x` is contiguous BF16 [K,T], `w` has logical shape [N,K], and `out` is contiguous BF16
 * [N,T], with the first index stored fastest. Registered execution uses RowSplit Q4G64_F16S,
 * Q5G64_F16S, Q6G64_F16S, or W8G32_F16S weights with FP16 scales. Each format owns a finite
 * registry of exact physical weight problems and selects its kernel internally; the encoding and
 * alignment contract alone do not imply arbitrary N/K support. Text/MTP problems accept every
 * positive token extent T. Registered Vision problems instead accept raw-patch
 * P in {4,8,...,131072} or merged-token V in [1,32768]; a matrix column is not automatically token
 * T. BF16_CTRL has a reserved
 * format-local boundary but currently admits no pure Linear problem. FP32_CTRL is unsupported by
 * this Op. The oracle exact-decodes the weight and evaluates each dot product naively in FP64
 * before converting the observable output to BF16. Production accumulator, staging, and reduction
 * choices are route-private. `out` must not overlap x or any weight plane. `ws` is caller-owned
 * transient storage and carries no semantic state beyond the call.
 */
void linear(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops
