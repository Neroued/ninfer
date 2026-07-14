#pragma once

// ninfer::kernels - fused GDN Q/K/V input projections into one contiguous output.

#include "ninfer/core/arena.h"
#include "ninfer/core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels {

void gdn_input_proj(const Tensor& x, const Weight& qk_weight, const Weight& v_weight, Tensor& qkv,
                    WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::kernels
