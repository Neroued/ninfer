#pragma once

// ninfer::kernels::detail - private launch prototype for argmax.

#include "ninfer/core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels::detail {

void argmax_launch(const Tensor& logits, Tensor& out, cudaStream_t stream);

} // namespace ninfer::kernels::detail
