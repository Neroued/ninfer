#pragma once

// qus::kernels - rmsnorm over the fastest dimension ne[0].
// unit_offset=true computes x/rms(x) * (1 + weight); false uses plain weight.
// If z is present, output is additionally multiplied by SiLU(z) after normalization.

#include "qus/core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace qus::kernels {

void rmsnorm(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, const Tensor* z,
             Tensor& out, cudaStream_t stream);

} // namespace qus::kernels
