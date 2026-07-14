#pragma once

// In-place Qwen3.6 RoPE. Dispatches Text 1-D [T], Text MRoPE [T,3], and
// Vision 2-D [T,2] from position rank and fixed model tensor shapes.

#include "ninfer/core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::kernels {

void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          cudaStream_t stream);

// Single-tensor form. Q/K role is inferred from the head-count shape.
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x, cudaStream_t stream);

} // namespace ninfer::kernels
