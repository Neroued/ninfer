#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels::detail {

void linear_rowsplit_gemv_gdn_in_qk_4096_q4_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                   WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::kernels::detail
