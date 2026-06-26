#pragma once

// qus::kernels::detail - private launch prototype for argmax.

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels::detail {

void argmax_launch(const Tensor& logits, Tensor& out, cudaStream_t stream);

} // namespace qus::kernels::detail
