#pragma once

// ninfer::kernels - sigmoid_mul kernel: x *= sigmoid(gate), elementwise in place.
// sigmoid(x) = 1 / (1 + e^-x), computed in fp32 with expf (NOT a polynomial fit).
// Vectorized over bf16 pairs; included only by its launcher. See
// docs/kernel-development.md §6.

#include "kernels/common/math.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::kernels {

inline constexpr int kSigmoidGateMulPairsPerThread = 4;

__device__ __forceinline__ __nv_bfloat162 sigmoid_gate_mul_pair(__nv_bfloat162 gate,
                                                                __nv_bfloat162 x) {
    const float r0 = __low2float(x) * sigmoid(__low2float(gate));
    const float r1 = __high2float(x) * sigmoid(__high2float(gate));
    return __floats2bfloat162_rn(r0, r1);
}

__global__ void sigmoid_gate_mul_scalar_kernel(const __nv_bfloat16* gate, __nv_bfloat16* x,
                                               std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        x[i] = __float2bfloat16_rn(__bfloat162float(x[i]) * sigmoid(__bfloat162float(gate[i])));
    }
}

__launch_bounds__(256) __global__
    void sigmoid_gate_mul_kernel(const __nv_bfloat16* gate, __nv_bfloat16* x, std::int64_t n) {
    const std::int64_t tid = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride =
        static_cast<std::int64_t>(gridDim.x) * blockDim.x * kSigmoidGateMulPairsPerThread;
    const std::int64_t n2 = n / 2;

    const auto* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate);
    auto* x2          = reinterpret_cast<__nv_bfloat162*>(x);
    for (std::int64_t j = tid * kSigmoidGateMulPairsPerThread; j < n2; j += stride) {
        const __nv_bfloat162 g0 = gate2[j];
        const __nv_bfloat162 x0 = x2[j];
        if (j + 3 < n2) {
            const __nv_bfloat162 g1  = gate2[j + 1];
            const __nv_bfloat162 x1  = x2[j + 1];
            const __nv_bfloat162 g2  = gate2[j + 2];
            const __nv_bfloat162 x2v = x2[j + 2];
            const __nv_bfloat162 g3  = gate2[j + 3];
            const __nv_bfloat162 x3  = x2[j + 3];
            x2[j]                    = sigmoid_gate_mul_pair(g0, x0);
            x2[j + 1]                = sigmoid_gate_mul_pair(g1, x1);
            x2[j + 2]                = sigmoid_gate_mul_pair(g2, x2v);
            x2[j + 3]                = sigmoid_gate_mul_pair(g3, x3);
        } else {
            x2[j] = sigmoid_gate_mul_pair(g0, x0);
            if (j + 1 < n2) { x2[j + 1] = sigmoid_gate_mul_pair(gate2[j + 1], x2[j + 1]); }
            if (j + 2 < n2) { x2[j + 2] = sigmoid_gate_mul_pair(gate2[j + 2], x2[j + 2]); }
        }
    }

    if (tid == 0 && (n & 1) != 0) {
        const std::int64_t i = n - 1;
        x[i] = __float2bfloat16_rn(__bfloat162float(x[i]) * sigmoid(__bfloat162float(gate[i])));
    }
}

} // namespace ninfer::kernels
