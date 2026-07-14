#pragma once

// ninfer::kernels - fused gate/up projection followed by SwiGLU.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels {

void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, WorkspaceArena& ws,
                   cudaStream_t stream);

} // namespace ninfer::kernels
