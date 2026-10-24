#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail::linear_bias {

using Launch = void (*)(const Tensor&, const Weight&, const Tensor&, Tensor&, cudaStream_t);
Launch select_launch(int rows, int k, int columns);
Launch select_n1024_k1176(int columns);
Launch select_n1024_k1024(int columns);
Launch select_n1024_k4096(int columns);
Launch select_n4096_k4096(int columns);

} // namespace ninfer::ops::detail::linear_bias
