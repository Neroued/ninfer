#pragma once

#include "ninfer/core/tensor.h"
#include "ninfer/kernels/gelu.h"

#include <cuda_runtime.h>

namespace ninfer::kernels::detail {

void gelu_launch(Tensor& x, GeluMode mode, cudaStream_t stream);

} // namespace ninfer::kernels::detail
