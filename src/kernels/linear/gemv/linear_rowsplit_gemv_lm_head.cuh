#pragma once

#include "ninfer/core/arena.h"
#include "ninfer/core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels::detail {

void linear_rowsplit_gemv_lm_head_q6_launch(const Tensor& x, const Weight& w, Tensor& out,
                                            WorkspaceArena& ws, cudaStream_t stream);

void linear_rowsplit_gemv_lm_head_q4_launch(const Tensor& x, const Weight& w, Tensor& out,
                                            WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::kernels::detail
