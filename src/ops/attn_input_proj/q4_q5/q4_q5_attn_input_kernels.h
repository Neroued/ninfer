#pragma once

#include "core/tensor.h"
#include "ops/linear/q4/q4_rowsplit_launch.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q4_q5_attn_input_grouped_mma_launch(Q4KernelVariant variant, const Tensor& x,
                                         const Weight& q_weight, const Weight& gate_weight,
                                         const Weight& k_weight, const Weight& v_weight, Tensor& q,
                                         Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);

} // namespace ninfer::ops::detail
