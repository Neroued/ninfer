#pragma once

// qus::kernels — silu_and_mul kernel: out = silu(gate) * up, elementwise.
// silu(x) = x / (1 + e^-x), computed exactly in fp32 (NOT a polynomial fit).
// Vectorized over bf16 pairs; included only by its launcher. See
// docs/l1-kernel-layering.md §6 and docs/l1-op-test-standard.md §0 (no math approximation).

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels {

inline constexpr int kSiluAndMulPairsPerThread = 4;

__device__ __forceinline__ float silu_f32(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ __nv_bfloat162 silu_mul_pair(__nv_bfloat162 g, __nv_bfloat162 u) {
    const float r0 = silu_f32(__low2float(g)) * __low2float(u);
    const float r1 = silu_f32(__high2float(g)) * __high2float(u);
    return __floats2bfloat162_rn(r0, r1);
}

__global__ void silu_and_mul_scalar_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                           __nv_bfloat16* out, std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        out[i] = __float2bfloat16(silu_f32(__bfloat162float(gate[i])) * __bfloat162float(up[i]));
    }
}

__global__ void silu_and_mul_strided_input_kernel(
    const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out, std::int64_t n,
    std::int32_t ne0, std::int32_t ne1, std::int32_t ne2, std::int64_t gnb0, std::int64_t gnb1,
    std::int64_t gnb2, std::int64_t gnb3, std::int64_t unb0, std::int64_t unb1,
    std::int64_t unb2, std::int64_t unb3) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const auto* gate_bytes    = reinterpret_cast<const unsigned char*>(gate);
    const auto* up_bytes      = reinterpret_cast<const unsigned char*>(up);
    for (std::int64_t i = start; i < n; i += stride) {
        std::int64_t rem = i;
        const auto d0    = static_cast<std::int32_t>(rem % ne0);
        rem /= ne0;
        const auto d1 = static_cast<std::int32_t>(rem % ne1);
        rem /= ne1;
        const auto d2 = static_cast<std::int32_t>(rem % ne2);
        const auto d3 = static_cast<std::int32_t>(rem / ne2);

        const std::int64_t goff = static_cast<std::int64_t>(d0) * gnb0 +
                                  static_cast<std::int64_t>(d1) * gnb1 +
                                  static_cast<std::int64_t>(d2) * gnb2 +
                                  static_cast<std::int64_t>(d3) * gnb3;
        const std::int64_t uoff = static_cast<std::int64_t>(d0) * unb0 +
                                  static_cast<std::int64_t>(d1) * unb1 +
                                  static_cast<std::int64_t>(d2) * unb2 +
                                  static_cast<std::int64_t>(d3) * unb3;
        const auto gv = *reinterpret_cast<const __nv_bfloat16*>(gate_bytes + goff);
        const auto uv = *reinterpret_cast<const __nv_bfloat16*>(up_bytes + uoff);
        out[i]        = __float2bfloat16(silu_f32(__bfloat162float(gv)) * __bfloat162float(uv));
    }
}

__launch_bounds__(256) __global__
    void silu_and_mul_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out,
                             std::int64_t n) {
    const std::int64_t tid = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride =
        static_cast<std::int64_t>(gridDim.x) * blockDim.x * kSiluAndMulPairsPerThread;
    const std::int64_t n2 = n / 2;

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
            if (j + 1 < n2) { out2[j + 1] = silu_mul_pair(gate2[j + 1], up2[j + 1]); }
            if (j + 2 < n2) { out2[j + 2] = silu_mul_pair(gate2[j + 2], up2[j + 2]); }
        }
    }

    if (tid == 0) {
        if ((n & 1) != 0) {
            const std::int64_t i = n - 1;
            out[i] =
                __float2bfloat16(silu_f32(__bfloat162float(gate[i])) * __bfloat162float(up[i]));
        }
    }
}

} // namespace qus::kernels
