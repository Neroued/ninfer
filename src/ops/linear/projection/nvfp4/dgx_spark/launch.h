#pragma once
#include "ops/linear/kernels/nvfp4/nvfp4_a4_plan.h"

namespace ninfer::ops::detail {
void launch_nvfp4_n12288_k4096_tma128(const Weight&, Tensor&, Nvfp4A4Workspace, int, cudaStream_t);
void launch_nvfp4_n12288_k4096_tma256(const Weight&, Tensor&, Nvfp4A4Workspace, int, cudaStream_t);
void launch_nvfp4_n4096_k6144_tma128(const Weight&, Tensor&, Nvfp4A4Workspace, int, cudaStream_t);
void launch_nvfp4_n4096_k6144_tma256(const Weight&, Tensor&, Nvfp4A4Workspace, int, cudaStream_t);
} // namespace ninfer::ops::detail
