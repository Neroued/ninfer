#pragma once
#include "core/weight.h"
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using LinearPairLaunch = void (*)(const Tensor&, const Weight&, const Weight&, Tensor&, Tensor&,
                                  cudaStream_t);
using Bf16PairLaunch = void (*)(const Tensor&, const Tensor&, const Weight&, const Weight&, Tensor&,
                                Tensor&, cudaStream_t);

LinearPairLaunch select_q8_pair_n1024_k2048(int tokens);
LinearPairLaunch select_q8_pair_n1024_k5120(int tokens);
LinearPairLaunch select_linear_pair(int k, int tokens);
void launch_bf16_pair_k128(const Tensor&, const Tensor&, const Weight&, const Weight&, Tensor&,
                           Tensor&, cudaStream_t);
void launch_bf16_pair_n32_k2304(const Tensor&, const Tensor&, const Weight&, const Weight&,
                                Tensor&, Tensor&, cudaStream_t);
void launch_bf16_pair_n1024_k2304(const Tensor&, const Tensor&, const Weight&, const Weight&,
                                  Tensor&, Tensor&, cudaStream_t);
Bf16PairLaunch validate_bf16_pair_geometry(int n, int k, int max_tokens);
// Returns the registered BF16 launcher after validation, or nullptr for a Q8 pair.
Bf16PairLaunch validate_pair_weights(const Weight& first, const Weight& second, int max_tokens);
} // namespace ninfer::ops::detail
