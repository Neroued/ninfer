#pragma once

// ninfer::ops::detail - private launch prototype for argmax.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void argmax_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows, cudaStream_t stream);
void argmax_tiled_atomic_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
                                int block, cudaStream_t stream);

} // namespace ninfer::ops::detail
