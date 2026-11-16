#pragma once

#include "ops/common/numeric/math.h"

#include <cuda_runtime.h>

namespace ninfer::ops::common {

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
