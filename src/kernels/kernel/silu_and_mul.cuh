#pragma once

// qus::kernels — silu_and_mul kernel: out = silu(gate) * up, elementwise.
// silu(x) = x / (1 + e^-x), computed exactly in fp32 (NOT a polynomial fit).
// Vectorized over bf16 pairs; included only by its launcher. See
// docs/l1-kernel-layering.md §6 and docs/l1-op-test-standard.md §0 (no math approximation).

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels {

inline constexpr int kSiluAndMulPairsPerThread = 4;

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ __nv_bfloat162 silu_mul_pair(__nv_bfloat162 g, __nv_bfloat162 u) {
    const float r0 = silu_f32(__low2float(g)) * __low2float(u);
    const float r1 = silu_f32(__high2float(g)) * __high2float(u);
    return __floats2bfloat162_rn(r0, r1);
}

__global__ void silu_and_mul_scalar_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                           __nv_bfloat16* out, std::int64_t n) {
    const std::int64_t start =
        blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        out[i] = __float2bfloat16(silu_f32(__bfloat162float(gate[i])) * __bfloat162float(up[i]));
    }
}

__launch_bounds__(256) __global__ void silu_and_mul_kernel(const __nv_bfloat16* gate,
                                                           const __nv_bfloat16* up,
                                                           __nv_bfloat16* out, std::int64_t n) {
    const std::int64_t tid = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride =
        static_cast<std::int64_t>(gridDim.x) * blockDim.x * kSiluAndMulPairsPerThread;
    const std::int64_t n2     = n / 2;

    const auto* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate);
    const auto* up2   = reinterpret_cast<const __nv_bfloat162*>(up);
    auto* out2        = reinterpret_cast<__nv_bfloat162*>(out);
    for (std::int64_t j = tid * kSiluAndMulPairsPerThread; j < n2; j += stride) {
        const __nv_bfloat162 g0 = gate2[j];
        const __nv_bfloat162 u0 = up2[j];
        if (j + 3 < n2) {
            const __nv_bfloat162 g1 = gate2[j + 1];
            const __nv_bfloat162 u1 = up2[j + 1];
            const __nv_bfloat162 g2 = gate2[j + 2];
            const __nv_bfloat162 u2 = up2[j + 2];
            const __nv_bfloat162 g3 = gate2[j + 3];
            const __nv_bfloat162 u3 = up2[j + 3];
            out2[j]                 = silu_mul_pair(g0, u0);
            out2[j + 1]             = silu_mul_pair(g1, u1);
            out2[j + 2]             = silu_mul_pair(g2, u2);
            out2[j + 3]             = silu_mul_pair(g3, u3);
        } else {
            out2[j] = silu_mul_pair(g0, u0);
            if (j + 1 < n2) {
                out2[j + 1] = silu_mul_pair(gate2[j + 1], up2[j + 1]);
            }
            if (j + 2 < n2) {
                out2[j + 2] = silu_mul_pair(gate2[j + 2], up2[j + 2]);
            }
        }
    }

    if (tid == 0) {
        if ((n & 1) != 0) {
            const std::int64_t i = n - 1;
            out[i] = __float2bfloat16(silu_f32(__bfloat162float(gate[i])) * __bfloat162float(up[i]));
        }
    }
}

} // namespace qus::kernels
