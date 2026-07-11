#pragma once

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels {

// Affine LayerNorm over ne[0]. Mean/variance and affine math use FP32; output is BF16.
void layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps, Tensor& out,
                cudaStream_t stream);

} // namespace qus::kernels
