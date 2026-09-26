#pragma once

#include "ops/common/math.h"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

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

__device__ __forceinline__ std::uint32_t pack_bf16x2(float lo, float hi) {
    std::uint32_t out;
    const std::uint32_t lo_bits = __float_as_uint(lo);
    const std::uint32_t hi_bits = __float_as_uint(hi);
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(out) : "r"(hi_bits), "r"(lo_bits));
    return out;
}

// Exact BF16 expansion for consumers that need FP32 operand bits, including TF32 MMA.
__device__ __forceinline__ void unpack_bf16x2_to_fp32_bits(unsigned packed, unsigned& low,
                                                           unsigned& high) {
    low  = packed << 16;
    high = packed & 0xffff0000U;
}

__device__ __forceinline__ std::uint32_t pack_f16x2(float lo, float hi) {
    const __half2 packed = __floats2half2_rn(lo, hi);
    return load_vec<std::uint32_t>(&packed);
}

__device__ __forceinline__ std::uint32_t bf16x2_bits_to_f16x2_bits(std::uint32_t bits) {
    const __nv_bfloat162 source = load_vec<__nv_bfloat162>(&bits);
    const __half2 converted =
        __halves2half2(__half(__low2bfloat16(source)), __half(__high2bfloat16(source)));
    return load_vec<std::uint32_t>(&converted);
}

__device__ __forceinline__ int4 bf16x8_bits_to_f16x8_bits(int4 bits) {
    return make_int4(
        static_cast<int>(bf16x2_bits_to_f16x2_bits(static_cast<std::uint32_t>(bits.x))),
        static_cast<int>(bf16x2_bits_to_f16x2_bits(static_cast<std::uint32_t>(bits.y))),
        static_cast<int>(bf16x2_bits_to_f16x2_bits(static_cast<std::uint32_t>(bits.z))),
        static_cast<int>(bf16x2_bits_to_f16x2_bits(static_cast<std::uint32_t>(bits.w))));
}

__device__ __forceinline__ float2 bf16x2_to_float2(__nv_bfloat162 value) {
    return __bfloat1622float2(value);
}

__device__ __forceinline__ float2 bf16x2_bits_to_float2(std::uint32_t bits) {
    return bf16x2_to_float2(load_vec<__nv_bfloat162>(&bits));
}

__device__ __forceinline__ void load_bf16x4(float (&values)[4], const __nv_bfloat16* source) {
    // The source must be 8-byte aligned. A built-in vector keeps one 64-bit load;
    // loading a pair-of-pairs struct can scalarize into two 32-bit requests.
    const uint2 packed = load_vec<uint2>(source);
    const float2 low   = bf16x2_bits_to_float2(packed.x);
    const float2 high  = bf16x2_bits_to_float2(packed.y);
    values[0]          = low.x;
    values[1]          = low.y;
    values[2]          = high.x;
    values[3]          = high.y;
}

__device__ __forceinline__ __half2 half2_from_bits(std::uint32_t bits) {
    return load_vec<__half2>(&bits);
}

} // namespace ninfer::ops
