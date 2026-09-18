#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

enum class Q8DynamicConvAddSchedule { TiledMma, MmaK128 };

void q8_dynamic_grouped_conv_add_materialized_launch(Q8DynamicConvAddSchedule schedule,
                                                     const Tensor& x, const Weight& weight,
                                                     const Tensor& base_kernel,
                                                     const Tensor& finish_delta, Tensor& residual,
                                                     Tensor& projected, cudaStream_t stream);

} // namespace ninfer::ops::detail
