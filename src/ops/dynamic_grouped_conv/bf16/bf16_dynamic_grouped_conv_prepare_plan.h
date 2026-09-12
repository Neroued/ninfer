#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

std::size_t bf16_dynamic_grouped_conv_prepare_workspace_capacity_bytes(std::int32_t min_width,
                                                                       std::int32_t max_width,
                                                                       std::int32_t min_batch_size,
                                                                       std::int32_t max_batch_size);

void bf16_dynamic_grouped_conv_prepare_dispatch(const Tensor& residual, const Tensor& norm_weight,
                                                float eps, const Tensor& base_kernel,
                                                const Weight& kernel_projection_weight,
                                                Tensor& prepared, Tensor& finish_delta,
                                                WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
