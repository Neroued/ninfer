#pragma once

// Split-projection FP8 SwiGLU on Volta tensor cores (sm_70 only). The FP8 sibling of
// nvfp4_linear_swiglu_qpn_split.cuh -- same reasoning throughout (see that file for the full
// argument: the fused kernel's doubled accumulator and decode registers cost more than sharing
// the activation load saves, and fp32 scratch is what the precision fix needed anyway) -- and
// simpler to address: FP8's scale is a flat N-length BF16 array with no swizzle at all, so the up
// half's scale pointer is just `scales + kIntermediate` elements, no offset-tile arithmetic.

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/linear/fp8/fp8_output.cuh"
#include "ops/linear/fp8/fp8_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

// Reads two fp32 [kIntermediate, T] planes (token-major, matching Fp8Fp32ContiguousOutput's store
// layout) and writes silu(gate) * up as BF16. Output-sized traffic, not weight-sized.
__global__ void fp8_swiglu_fp32_combine_kernel(const float* __restrict__ gate,
                                                const float* __restrict__ up,
                                                __nv_bfloat16* __restrict__ out, std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        out[i] = __float2bfloat16_rn(silu(gate[i]) * up[i]);
    }
}

#endif // sm_70

} // namespace ninfer::ops::detail
