#pragma once

#include "ops/common/math.h"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstring>

namespace ninfer::ops {

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float softplus(float x) { return (x > 20.0f) ? x : log1pf(expf(x)); }

__device__ __forceinline__ float exp2_approx(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// cvt.rn.bf16x2.f32 is Ampere+ only (like cp.async, this was missing from the original
// full-tree audit — see docs/v100.md). Below sm_80, __floats2bfloat162_rn is the
// portable equivalent: a standard cuda_bf16.h intrinsic (software round-to-nearest-even
// on pre-Ampere, not a hardware instruction), already used elsewhere in this codebase
// (e.g. Q4MmaDecodeAtom::decode_pair) for exactly this reason.
__device__ __forceinline__ std::uint32_t pack_bf16x2(float lo, float hi) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    std::uint32_t out;
    const std::uint32_t lo_bits = __float_as_uint(lo);
    const std::uint32_t hi_bits = __float_as_uint(hi);
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(out) : "r"(hi_bits), "r"(lo_bits));
    return out;
#else
    const __nv_bfloat162 packed = __floats2bfloat162_rn(lo, hi);
    std::uint32_t out;
    std::memcpy(&out, &packed, sizeof(out));
    return out;
#endif
}

__device__ __forceinline__ float2 bf16x2_to_float2(__nv_bfloat162 value) {
    return __bfloat1622float2(value);
}

__device__ __forceinline__ float2 bf16x2_bits_to_float2(std::uint32_t bits) {
    return bf16x2_to_float2(load_vec<__nv_bfloat162>(&bits));
}

__device__ __forceinline__ __half2 half2_from_bits(std::uint32_t bits) {
    return load_vec<__half2>(&bits);
}

} // namespace ninfer::ops
