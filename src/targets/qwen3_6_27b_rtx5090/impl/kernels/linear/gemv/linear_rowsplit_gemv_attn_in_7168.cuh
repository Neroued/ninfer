#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels::detail {

void linear_rowsplit_gemv_attn_in_7168_q4_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                 WorkspaceArena& ws, cudaStream_t stream);
void linear_rowsplit_gemv_attn_in_7168_q5_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                 WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::kernels::detail
