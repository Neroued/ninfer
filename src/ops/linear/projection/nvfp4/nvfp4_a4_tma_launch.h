#pragma once
#include "ops/linear/kernels/nvfp4/nvfp4_operands.h"

namespace ninfer::ops::detail {
inline constexpr int kNvfp4LinearTmaKPartitions = 2;
void launch_nvfp4_a4_tma_linear(Nvfp4GeometryId problem, const Nvfp4A4Operands& p,
                                __nv_bfloat16* output, cudaStream_t stream);
void launch_nvfp4_a4_tma_linear_split(Nvfp4GeometryId problem, const Nvfp4A4Operands& p,
                                      Nvfp4A4Workspace workspace, __nv_bfloat16* output,
                                      cudaStream_t stream);
} // namespace ninfer::ops::detail
