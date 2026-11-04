#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

namespace ninfer::ops::detail {

using Nvfp4MulBiasLaunch = void (*)(const Tensor&, const Weight&, const Tensor&, Tensor&,
                                    cudaStream_t);

Nvfp4MulBiasLaunch select_nvfp4_mul_bias_launch(int rows, int k, int tokens, LinearPolicy);
Nvfp4MulBiasLaunch select_nvfp4_mul_bias_n1024_k4096(int tokens);

std::size_t nvfp4_mul_bias_workspace_capacity_bytes(int rows, int k, LinearPolicy, int first,
                                                    int last);
void nvfp4_mul_bias_dispatch(const Tensor&, const Weight&, const Tensor&, Tensor&, LinearPolicy,
                             cudaStream_t);

} // namespace ninfer::ops::detail
