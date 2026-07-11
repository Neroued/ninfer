#pragma once

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels::detail {

void vision_pos_embed_add_launch(const Tensor& table, const Tensor& indices, const Tensor& weights,
                                 Tensor& x, cudaStream_t stream);

} // namespace qus::kernels::detail
