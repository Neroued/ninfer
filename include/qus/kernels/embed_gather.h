#pragma once

// qus::kernels - embed_gather: out[:, t] = table[ids[t], :].
// Precision seam op: BF16_CTRL copies dense rows through as_dense(table);
// Q6G64_F16S dequantizes RowGroupedG64 payload rows.

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels {

void embed_gather(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream);

} // namespace qus::kernels
