#pragma once

// ninfer::kernels::detail - private launch prototype for argmax.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels::detail {

void argmax_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
                   cudaStream_t stream);

} // namespace ninfer::kernels::detail
