#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t fp8_gdn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                                 std::int32_t min_tokens,
                                                                 std::int32_t max_tokens);

void fp8_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 cudaStream_t stream);

void fp8_gdn_input_matrix_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                  cudaStream_t stream);

void fp8_gdn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                             Fp8A8Workspace workspace, cudaStream_t stream);

// Exact contraction mechanisms shared by G1/G2/G3. Semantic Ops own their route frontier and
// call one of these launchers after resolving their complete-form plan.
void fp8_gdn_input_a16_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream);

void fp8_gdn_input_a8_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                               WorkspaceArena& workspace, cudaStream_t stream);

void fp8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                            LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
