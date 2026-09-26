#pragma once
#include "ops/linear/nvfp4/nvfp4_operands.h"

namespace ninfer::ops::detail {
void launch_nvfp4_linear_swiglu_a4_tma(const Nvfp4A4Operands& p, __nv_bfloat16* output,
                                       cudaStream_t stream);
} // namespace ninfer::ops::detail
