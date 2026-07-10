#pragma once

// qus::kernels - partial NeoX RoPE over q=[256,24,T] and k=[256,4,T], in place.
// Rotates the first rotary_dim channels and leaves remaining head dimensions untouched.

#include "qus/core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace qus::kernels {

void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          cudaStream_t stream);

// Single-tensor variants used when prompt KV preparation only materializes K, or when a final-row
// attention tail only materializes Q. They use the same partial NeoX rotation and arithmetic as
// rope(), without requiring an unused companion tensor.
void rope_q(const Tensor& positions, int rotary_dim, float theta, Tensor& q, cudaStream_t stream);
void rope_k(const Tensor& positions, int rotary_dim, float theta, Tensor& k, cudaStream_t stream);

} // namespace qus::kernels
