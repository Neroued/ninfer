#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                                    std::int32_t input_rows,
                                                                    LinearPolicy policy,
                                                                    std::int32_t min_tokens,
                                                                    std::int32_t max_tokens);

void nvfp4_linear_add_a16_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                 cudaStream_t stream);

void nvfp4_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                    cudaStream_t stream);

void nvfp4_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                     cudaStream_t stream);

void nvfp4_linear_add_a4_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                Nvfp4A4Workspace workspace, cudaStream_t stream);

void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                               LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
