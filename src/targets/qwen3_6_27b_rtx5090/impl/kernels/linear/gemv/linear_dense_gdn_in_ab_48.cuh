#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::kernels::detail {

std::size_t linear_dense_gdn_in_ab_gated_48_workspace_bytes(std::int32_t tokens);

void linear_dense_gdn_in_ab_gated_48_launch(const Tensor& x, const Weight& a_weight,
                                            const Weight& b_weight, const Tensor& A_log,
                                            const Tensor& dt_bias, void* workspace,
                                            std::size_t workspace_bytes, Tensor& g, Tensor& beta,
                                            cudaStream_t stream);

} // namespace ninfer::kernels::detail
