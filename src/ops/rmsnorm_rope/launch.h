#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void rmsnorm_rope_pair_launch(const Tensor& positions, const Tensor& q_norm_weight,
                              const Tensor& k_norm_weight, Tensor& q, Tensor& k,
                              std::int32_t tokens, cudaStream_t stream);

void rmsnorm_rope_single_launch(const Tensor& positions, const Tensor& norm_weight, Tensor& x,
                                std::int32_t tokens, cudaStream_t stream);

} // namespace ninfer::ops::detail
