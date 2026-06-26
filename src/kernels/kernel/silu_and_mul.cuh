#pragma once

// qus::kernels — silu_and_mul kernel: out = silu(gate) * up, elementwise.
// Correctness-baseline form (grid-stride, fp32 math); included only by its launcher.
// See docs/l1-kernel-layering.md §6.

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels {

__global__ void silu_and_mul_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                    __nv_bfloat16* out, std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const float g    = __bfloat162float(gate[i]);
        const float u    = __bfloat162float(up[i]);
        const float silu = g / (1.0f + expf(-g));
        out[i]           = __float2bfloat16(silu * u);
    }
}

} // namespace qus::kernels
