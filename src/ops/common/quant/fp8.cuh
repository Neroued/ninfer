#pragma once

#include "ops/common/cuda/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops::common {

inline constexpr float kFp8E4M3MaxFinite = 448.0F;

// Dynamic E4M3 activation groups retain their scale in FP32.
__device__ __forceinline__ float fp8_fp32_scale(float absmax) {
    return __fdiv_rn(absmax, kFp8E4M3MaxFinite);
}

__device__ __forceinline__ std::uint16_t fp8_quant_fp32_pair(float2 value, float scale) {
    if (scale == 0.0F) { return 0; }
    return __nv_cvt_float2_to_fp8x2(
        make_float2(__fdiv_rn(value.x, scale), __fdiv_rn(value.y, scale)), __NV_SATFINITE,
        __NV_E4M3);
}

__device__ __forceinline__ unsigned fp8_e4m3x2_to_bf16x2_bits(unsigned packed) {
#if __CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)
    // PTX 9.2 exposes the native pair widening on the project's sm_120a target.
    unsigned result;
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(result) : "h"(static_cast<std::uint16_t>(packed)));
    return result;
#else
    // CUDA 13.1 retains the exact FP16 bridge.
    __nv_fp8x2_e4m3 fp8;
    fp8.__x                 = static_cast<std::uint16_t>(packed);
    const __half2 half_pair = static_cast<__half2>(fp8);

    union {
        __nv_bfloat162 pair;
        unsigned bits;
    } result;

    result.pair = __halves2bfloat162(__nv_bfloat16(__low2half(half_pair)),
                                     __nv_bfloat16(__high2half(half_pair)));
    return result.bits;
#endif
}

__device__ __forceinline__ float decode_e4m3(std::uint8_t storage) {
    __nv_fp8x2_e4m3 value;
    value.__x = static_cast<std::uint16_t>(storage) | (static_cast<std::uint16_t>(storage) << 8);
    return static_cast<float2>(value).x;
}

__device__ __forceinline__ float2 fp8_code2_to_float2(std::uint16_t storage) {
    __nv_fp8x2_e4m3 value;
    value.__x = storage;
    return static_cast<float2>(value);
}

// UE8M0 is a positive binary exponent, biased by 127; byte zero is 2^-127.
__device__ __forceinline__ float ue8m0_to_float(std::uint8_t scale) {
    return __uint_as_float(scale ? unsigned(scale) << 23 : 0x00400000u);
}

struct Fp8Ue8m0Scale {
    std::uint8_t scale;
    float inverse_scale;
};

__device__ __forceinline__ Fp8Ue8m0Scale fp8_ue8m0_scale(float absmax) {
    if (absmax == 0.0F) return {127, 1.0F};
    // 448 = 1.75 * 2^8. Compare the significand before division to round the
    // exponent upward without log2 or rounding amax/448 at a power-of-two seam.
    const unsigned bits = __float_as_uint(absmax);
    const int exponent  = max(0, int(bits >> 23) - 8 + int((bits & 0x7fffffu) > 0x600000u));
    return {static_cast<std::uint8_t>(exponent), __uint_as_float(unsigned(254 - exponent) << 23)};
}

__device__ __forceinline__ std::uint8_t fp8_quant_code(float x, float inverse_scale) {
    if (inverse_scale == 0.0F) { return 0; }
    return __nv_cvt_float_to_fp8(x * inverse_scale, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ std::uint16_t fp8_quant_code2(float x0, float x1, float inverse_scale) {
    if (inverse_scale == 0.0F) { return 0; }
    return __nv_cvt_float2_to_fp8x2(make_float2(x0 * inverse_scale, x1 * inverse_scale),
                                    __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ __half2 fp8_code2_to_half2(std::uint16_t storage) {
    __nv_fp8x2_e4m3 value;
    value.__x = storage;
    return static_cast<__half2>(value);
}

// An already decoded FP16 multiplier, shared by all values in one G128 group.
__device__ __forceinline__ int4 fp8_dequant_f16x8_scaled(const std::uint8_t* codes, __half scale) {
    const int2 raw    = load_vec<int2>(codes);
    const auto* c     = reinterpret_cast<const std::uint16_t*>(&raw);
    const auto scale2 = __halves2half2(scale, scale);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2 value = __hmul2(fp8_code2_to_half2(c[i]), scale2);
        packed[i]           = *reinterpret_cast<const unsigned*>(&value);
    }
    return make_int4(packed[0], packed[1], packed[2], packed[3]);
}

// Full UE8M0 range: the first power-of-two multiply is exact for finite E4M3
// results, and the residual multiplier performs the single final FP16 rounding.
__device__ __forceinline__ int4 fp8_dequant_f16x8(const std::uint8_t* codes, std::uint8_t scale) {
    const int primary = min(142, max(113, int(scale)));
    const auto first  = __ushort_as_half(static_cast<unsigned short>((primary - 112) << 10));
    int4 result       = fp8_dequant_f16x8_scaled(codes, first);
    if (int(scale) != primary) {
        const int residual    = min(15, max(-24, int(scale) - primary));
        const auto second     = __float2half_rn(__uint_as_float(unsigned(residual + 127) << 23));
        const auto multiplier = __halves2half2(second, second);
        auto* pairs           = reinterpret_cast<__half2*>(&result);
#pragma unroll
        for (int i = 0; i < 4; ++i) pairs[i] = __hmul2(pairs[i], multiplier);
    }
    return result;
}

__device__ __forceinline__ unsigned fp8_dequant_bf16x2(std::uint16_t codes, std::uint8_t scale) {
    const unsigned decoded    = fp8_e4m3x2_to_bf16x2_bits(codes);
    const unsigned exponent   = scale ? scale << 7 : 0x40u;
    const unsigned multiplier = exponent | (exponent << 16);
    const auto value =
        __hmul2(load_vec<__nv_bfloat162>(&decoded), load_vec<__nv_bfloat162>(&multiplier));
    return load_vec<unsigned>(&value);
}

} // namespace ninfer::ops::common
