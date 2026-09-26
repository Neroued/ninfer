#pragma once

#include <cuda_runtime.h>

namespace ninfer::ops::detail::kimi_delta_attention {

inline constexpr float kLog2E = 1.4426950408889634F;

__device__ __forceinline__ float sigmoid_approx(float value) {
    float result;
    asm("tanh.approx.f32 %0, %1;" : "=f"(result) : "f"(0.5F * value));
    return 0.5F * result + 0.5F;
}

__device__ __forceinline__ float exp2_approx(float value) {
    float result;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ float exp_approx(float value) { return exp2_approx(value * kLog2E); }

} // namespace ninfer::ops::detail::kimi_delta_attention
