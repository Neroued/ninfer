#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void rmsnorm_pack_tail_launch(const Tensor& input, const Tensor& weight, Tensor& output,
                              cudaStream_t stream);

} // namespace ninfer::ops::detail
