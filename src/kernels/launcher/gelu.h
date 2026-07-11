#pragma once

#include "qus/core/tensor.h"
#include "qus/kernels/gelu.h"

#include <cuda_runtime.h>

namespace qus::kernels::detail {

void gelu_launch(Tensor& x, GeluMode mode, cudaStream_t stream);

} // namespace qus::kernels::detail
