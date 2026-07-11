#pragma once

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels {

enum class GeluMode {
    Exact,
    Tanh,
};

// In-place GELU. Both modes compute in FP32 and round once to BF16.
void gelu(Tensor& x, GeluMode mode, cudaStream_t stream);

} // namespace qus::kernels
