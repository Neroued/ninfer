#pragma once
#include "ops/linear/kernels/nvfp4/nvfp4_operands.h"

namespace ninfer::ops::detail {
#if NINFER_TARGET_RTX5090
void launch_nvfp4_swiglu_n28672_k5120_tma(const Nvfp4A4Operands& p, __nv_bfloat16* y,
                                           cudaStream_t stream);
#endif
void launch_nvfp4_swiglu_a4_tma(const Nvfp4A4Operands& p, __nv_bfloat16* output,
                                cudaStream_t stream);
} // namespace ninfer::ops::detail
