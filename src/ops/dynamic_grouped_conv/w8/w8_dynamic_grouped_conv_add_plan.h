#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

std::size_t w8_linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
    std::int32_t input_rows, std::int32_t min_width, std::int32_t max_width,
    std::int32_t min_batch_size, std::int32_t max_batch_size);

const char* w8_linear_dynamic_grouped_conv_add_route_name(std::int32_t input_rows,
                                                          std::int32_t width,
                                                          std::int32_t batch_size);

void w8_linear_dynamic_grouped_conv_add_dispatch(const Tensor& x, const Weight& projection_weight,
                                                 const Tensor& base_kernel,
                                                 const Tensor& finish_delta, Tensor& residual,
                                                 WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
