#pragma once

#include "ops/common/numeric/math.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::common {

// Exact division for numerator and positive divisor in [0, 2^31). The bounded
// numerator keeps the high-product addition representable in 32 bits.
struct FastDivisorU31 {
    std::uint32_t multiplier, shift, divisor;

    static FastDivisorU31 make(std::uint32_t divisor) {
        if (divisor == 0 || divisor >= (std::uint32_t{1} << 31))
            throw std::invalid_argument("FastDivisorU31: divisor must be in [1,2^31)");
        std::uint32_t shift = 0;
        while ((std::uint32_t{1} << shift) < divisor) ++shift;
        const auto multiplier = static_cast<std::uint32_t>(
            ((std::uint64_t{1} << 32) * ((std::uint64_t{1} << shift) - divisor)) / divisor + 1);
        return {multiplier, shift, divisor};
    }

    __host__ __device__ std::uint32_t divide(std::uint32_t numerator) const {
#if defined(__CUDA_ARCH__)
        return (__umulhi(numerator, multiplier) + numerator) >> shift;
#else
        return numerator / divisor;
#endif
    }
};

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// Fast tanh-based profile; saturation and rounding differ from sigmoid's expf path.
__device__ __forceinline__ float sigmoid_approx(float x) {
    float y;
    asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(0.5F * x));
    return 0.5F * y + 0.5F;
}

__device__ __forceinline__ float softplus(float x) { return (x > 20.0f) ? x : log1pf(expf(x)); }

__device__ __forceinline__ float exp2_approx(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// Keep flush-to-zero explicit so callers of exp2_approx retain their subnormal behavior.
__device__ __forceinline__ float exp2_approx_ftz(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float exp_approx_ftz(float x) { return exp2_approx_ftz(x * kLog2E); }

__device__ __forceinline__ float rcp_approx_ftz(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

template <bool TanhApprox>
__device__ __forceinline__ float gelu(float value) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    constexpr float kSqrt2Pi  = 0.79788456080286535588f;
    if constexpr (TanhApprox) {
        return 0.5f * value *
               (1.0f + tanhf(kSqrt2Pi * (value + 0.044715f * value * value * value)));
    }
    return 0.5f * value * (1.0f + erff(value * kInvSqrt2));
}

} // namespace ninfer::ops::common
