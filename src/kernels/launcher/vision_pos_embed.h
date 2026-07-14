#pragma once

#include "ninfer/core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels::detail {

void vision_pos_embed_add_launch(const Tensor& table, const Tensor& indices, const Tensor& weights,
                                 Tensor& x, cudaStream_t stream);

} // namespace ninfer::kernels::detail
