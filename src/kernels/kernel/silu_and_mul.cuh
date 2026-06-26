#pragma once

// qus::kernels — silu_and_mul kernel: out = silu(gate) * up, elementwise.
// silu(x) = x / (1 + e^-x), computed exactly in fp32 (NOT a polynomial fit).
// Vectorized over bf16 pairs; included only by its launcher. See
// docs/l1-kernel-layering.md §6 and docs/l1-op-test-standard.md §0 (no math approximation).

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels {

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + __expf(-x));
}

__launch_bounds__(1024) __global__ void silu_and_mul_kernel(const __nv_bfloat16* gate,
                                                            const __nv_bfloat16* up,
                                                            __nv_bfloat16* out, std::int64_t n) {
    const std::int64_t tid    = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t n2     = n / 2;

    const auto* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate);
    const auto* up2   = reinterpret_cast<const __nv_bfloat162*>(up);
    auto* out2        = reinterpret_cast<__nv_bfloat162*>(out);

    for (std::int64_t j = tid; j < n2; j += stride) {
        const __nv_bfloat162 g = gate2[j];
        const __nv_bfloat162 u = up2[j];
        const float r0 = silu_f32(__low2float(g)) * __low2float(u);
        const float r1 = silu_f32(__high2float(g)) * __high2float(u);
        out2[j]        = __floats2bfloat162_rn(r0, r1);
    }

    if ((n & 1) != 0 && tid == 0) {
        const std::int64_t i = n - 1;
        out[i] = __float2bfloat16(silu_f32(__bfloat162float(gate[i])) * __bfloat162float(up[i]));
    }
}

} // namespace qus::kernels
