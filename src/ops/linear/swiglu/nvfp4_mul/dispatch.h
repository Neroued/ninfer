#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

namespace ninfer::ops::detail {

using Nvfp4MulSwiGluLaunch = void (*)(const Tensor&, const Weight&, const Tensor*, Tensor&, float,
                                      bool, cudaStream_t);

Nvfp4MulSwiGluLaunch select_nvfp4_mul_swiglu_launch(int rows, int k, LinearPolicy);
void launch_nvfp4_mul_swiglu_n8192_k1024(const Tensor&, const Weight&, const Tensor*, Tensor&,
                                         float limit, bool clamped, cudaStream_t);
void launch_nvfp4_mul_swiglu_n20480_k4096(const Tensor&, const Weight&, const Tensor*, Tensor&,
                                          float limit, bool clamped, cudaStream_t);

std::size_t nvfp4_mul_swiglu_workspace_capacity_bytes(int rows, int k, LinearPolicy, int first,
                                                      int last);
void nvfp4_mul_swiglu_dispatch(const Tensor&, const Weight&, const Tensor*, Tensor&, float limit,
                               bool clamped, LinearPolicy, cudaStream_t);

} // namespace ninfer::ops::detail
