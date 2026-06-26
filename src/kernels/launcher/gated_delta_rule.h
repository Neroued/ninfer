#pragma once

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels::detail {

void gdn_cast_bf16_to_f32_launch(const Tensor& in, Tensor& out, cudaStream_t stream);
void gdn_cast_f32_to_bf16_launch(const Tensor& in, Tensor& out, cudaStream_t stream);

void gated_delta_rule_recurrent_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                       const Tensor& g, const Tensor& beta, float scale,
                                       Tensor& ssm_state, Tensor& out, cudaStream_t stream);

} // namespace qus::kernels::detail
