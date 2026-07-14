#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels::detail {

void linear_rowsplit_gemv_out_6144_q5_launch(const Tensor& x, const Weight& w, Tensor& out,
                                             WorkspaceArena& ws, cudaStream_t stream);

void linear_rowsplit_gemv_out_6144_residual_q5_launch(const Tensor& x, const Weight& w,
                                                      Tensor& residual_out, WorkspaceArena& ws,
                                                      cudaStream_t stream);

} // namespace ninfer::kernels::detail
