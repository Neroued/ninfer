#pragma once

// qus::kernels - partial NeoX RoPE over q=[256,24,T] and k=[256,4,T], in place.
// Rotates the first rotary_dim channels and leaves remaining head dimensions untouched.

#include "qus/core/tensor.h"

#include <cuda_runtime.h>  // cudaStream_t

namespace qus::kernels {

void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          cudaStream_t stream);

} // namespace qus::kernels
