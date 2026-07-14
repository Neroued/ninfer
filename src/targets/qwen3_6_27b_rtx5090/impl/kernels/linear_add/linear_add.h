#pragma once

// ninfer::kernels - fused residual += W @ x.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels {

void linear_add(const Tensor& x, const Weight& w, Tensor& residual, WorkspaceArena& ws,
                cudaStream_t stream);

} // namespace ninfer::kernels
