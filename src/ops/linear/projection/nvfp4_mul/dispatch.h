#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

namespace ninfer::ops::detail {

using Nvfp4MulLaunch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

struct Nvfp4MulRoute {
    Nvfp4MulLaunch launch;
    bool tensor_core;
};

Nvfp4MulRoute select_nvfp4_mul_route(int rows, int k, int tokens, LinearPolicy policy);
Nvfp4MulRoute select_nvfp4_mul_n8192_k1024(int tokens);
Nvfp4MulRoute select_nvfp4_mul_n1024_k4096(int tokens);
Nvfp4MulRoute select_nvfp4_mul_n4096_k4096(int tokens);
Nvfp4MulRoute select_nvfp4_mul_n20480_k4096(int tokens);
Nvfp4MulRoute select_nvfp4_mul_n4096_k10240(int tokens);

std::size_t nvfp4_mul_linear_workspace_capacity_bytes(int rows, int k, LinearPolicy policy,
                                                      int first, int last);
void nvfp4_mul_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy, cudaStream_t);

} // namespace ninfer::ops::detail
