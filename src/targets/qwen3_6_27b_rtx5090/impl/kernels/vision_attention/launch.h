#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels::detail {

void vision_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& cu_seqlens, Tensor* tiles, Tensor& out,
                             cudaStream_t stream);

} // namespace ninfer::kernels::detail
