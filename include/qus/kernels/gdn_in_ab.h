#pragma once

#include "qus/core/arena.h"
#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels {

void gdn_in_ab_gated(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                     const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws, Tensor& g,
                     Tensor& beta, cudaStream_t stream);

} // namespace qus::kernels
