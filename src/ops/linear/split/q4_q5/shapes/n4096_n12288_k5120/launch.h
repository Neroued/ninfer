#pragma once
#include "ops/linear/split/dispatch.h"

namespace ninfer::ops::detail::linear_split::q4_q5_n4096_n12288_k5120 {
void launch_small(const Tensor&, const Weight&, const Weight&, std::span<const Tensor>,
                  cudaStream_t);
void launch_matrix(const Tensor&, const Weight&, const Weight&, std::span<const Tensor>,
                   cudaStream_t);
} // namespace ninfer::ops::detail::linear_split::q4_q5_n4096_n12288_k5120
