#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using Q4LinearAddLaunch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

Q4LinearAddLaunch select_q4_linear_add(std::int32_t rows, std::int32_t k, std::int32_t tokens);

} // namespace ninfer::ops::detail
