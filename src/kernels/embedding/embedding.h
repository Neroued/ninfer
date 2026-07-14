#pragma once

// ninfer::kernels - embedding: out[:, t] = table[ids[t], :].
// Precision seam op: BF16_CTRL copies dense rows through as_dense(table);
// Q6G64_F16S dequantizes ROW_SPLIT nibble, high, and scale planes.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels {

void embedding(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream);

} // namespace ninfer::kernels
