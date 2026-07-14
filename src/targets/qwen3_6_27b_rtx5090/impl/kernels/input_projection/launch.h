#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels::detail {

void linear_rowsplit_attn_input_grouped_mma_launch(const Tensor& x, const Weight& q_weight,
                                                   const Weight& gate_weight,
                                                   const Weight& k_weight, const Weight& v_weight,
                                                   Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                                   cudaStream_t stream);

void linear_rowsplit_gdn_input_grouped_mma_launch(const Tensor& x, const Weight& qk_weight,
                                                  const Weight& v_weight, Tensor& qkv,
                                                  cudaStream_t stream);

} // namespace ninfer::kernels::detail
