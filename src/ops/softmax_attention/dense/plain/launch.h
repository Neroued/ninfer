#pragma once
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {
void launch_plain_attention(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& out,
                            cudaStream_t stream);
} // namespace ninfer::ops::detail
