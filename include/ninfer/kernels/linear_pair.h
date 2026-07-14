#pragma once

// ninfer::kernels - paired linear projection. Both outputs are W @ x projections;
// qtype, shape, and token-count dispatch remain internal.

#include "ninfer/core/arena.h"
#include "ninfer/core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels {

void linear_pair(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                 Tensor& first_out, Tensor& second_out, WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::kernels
