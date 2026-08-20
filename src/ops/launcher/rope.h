#pragma once

// ninfer::ops::detail - private launch prototype for rope. Included by the wrapper
// and defined by the CUDA launcher.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void rope_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                 cudaStream_t stream);

void rope_single_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                        cudaStream_t stream);

void qk_norm_rope_text_launch(const Tensor& positions, const Tensor& q_in, const Tensor& q_weight,
                              Tensor& q_out, const Tensor& k_in, const Tensor& k_weight,
                              Tensor& k_out, float eps, cudaStream_t stream);

} // namespace ninfer::ops::detail
