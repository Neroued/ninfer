#pragma once

#include "core/tensor.h"
#include "kernels/gelu/gelu.h"

#include <cuda_runtime.h>

namespace ninfer::kernels::detail {

void gelu_launch(Tensor& x, GeluMode mode, cudaStream_t stream);

} // namespace ninfer::kernels::detail
