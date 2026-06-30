#pragma once

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels {

void gdn_in_ab_decode(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                      Tensor& a_out, Tensor& b_out, cudaStream_t stream);

void gdn_in_ab_gated_decode(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                            const Tensor& A_log, const Tensor& dt_bias, Tensor& g, Tensor& beta,
                            cudaStream_t stream);

} // namespace qus::kernels
