#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

namespace ninfer::ops::detail {

using Fp8BlockSwiGluLaunch = void (*)(const Tensor&, const Weight&, Tensor&, float limit,
                                      bool clamped, cudaStream_t);
Fp8BlockSwiGluLaunch select_fp8_block_swiglu_launch(int rows, int k, int tokens, LinearPolicy);
Fp8BlockSwiGluLaunch select_fp8_block_swiglu_n2048_k4096(int tokens);
std::size_t fp8_block_swiglu_workspace_capacity_bytes(int rows, int k, LinearPolicy, int first,
                                                      int last);
void fp8_block_swiglu_dispatch(const Tensor&, const Weight&, Tensor&, float limit, bool clamped,
                               LinearPolicy, cudaStream_t);

} // namespace ninfer::ops::detail
