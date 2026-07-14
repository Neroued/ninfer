#pragma once

// ninfer::kernels::detail - private launch prototype for rope. Included by the wrapper
// and defined by the CUDA launcher.

#include "ninfer/core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels::detail {

void rope_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                 cudaStream_t stream);

void rope_q_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& q,
                   cudaStream_t stream);
void rope_k_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& k,
                   cudaStream_t stream);

void rope_nd_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                    cudaStream_t stream);
void rope_nd_single_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                           cudaStream_t stream);

} // namespace ninfer::kernels::detail
