#pragma once

#include "qus/core/arena.h"
#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels {

void mlp_gate_up_silu(const Tensor& x, const Weight& gate_up_weight, Tensor& out,
                      WorkspaceArena& ws, cudaStream_t stream);

} // namespace qus::kernels
