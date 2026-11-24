#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t fp8_linear_swiglu_partial_capacity_bytes(std::int32_t max_tokens);

[[nodiscard]] std::size_t fp8_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                                     std::int32_t min_tokens,
                                                                     std::int32_t max_tokens);

void launch_fp8_swiglu_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                               cudaStream_t stream);

void launch_fp8_swiglu_matrix(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_fp8_swiglu_a8(const Tensor& x, const Weight& weight, Tensor& out,
                          MemoryArena& workspace, cudaStream_t stream);

void fp8_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                LinearPolicy policy, MemoryArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
