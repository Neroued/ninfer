#pragma once

#include "ninfer/core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels {

// Four-corner interpolated position embedding followed by BF16 residual add.
// table: [D,table_rows], indices/weights: [4,P], x: [D,P].
void vision_pos_embed_add(const Tensor& table, const Tensor& indices, const Tensor& weights,
                          Tensor& x, cudaStream_t stream);

} // namespace ninfer::kernels
