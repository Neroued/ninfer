#pragma once
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {
struct DynamicConvPrepareRoute {
    int rows;
    int columns;
    int split_k;
};

void bf16_dynamic_grouped_conv_prepare_partial_launch(DynamicConvPrepareRoute route,
                                                      const Tensor& input, const Weight& weight,
                                                      float* partial, cudaStream_t stream);
void bf16_dynamic_grouped_conv_prepare_reduce_launch(DynamicConvPrepareRoute route,
                                                     const Tensor& base, const float* partial,
                                                     Tensor& prepared, Tensor& finish,
                                                     cudaStream_t stream);
} // namespace ninfer::ops::detail
