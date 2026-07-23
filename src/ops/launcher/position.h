#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void fill_i32_positions_launch(Tensor& positions, std::int32_t start, cudaStream_t stream);
void offset_i32_positions_launch(const Tensor& source, const Tensor& delta, Tensor& destination,
                                 cudaStream_t stream);
void offset_i32_positions_block_launch(const Tensor& source, const Tensor& delta,
                                       Tensor& destination, int block, cudaStream_t stream);

} // namespace ninfer::ops::detail
