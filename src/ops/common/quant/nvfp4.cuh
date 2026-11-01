#pragma once

#include "ops/common/quant/nvfp4.h"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/quant/fp8.cuh"

#include <cuda_fp4.h>

#include <cstdint>

namespace ninfer::ops::common {

__device__ __forceinline__ float2 decode_nvfp4_e2m1x2(std::uint8_t storage) {
    __nv_fp4x2_e2m1 value;
    value.__x = storage;
    return static_cast<float2>(value);
}

__device__ __forceinline__ unsigned nvfp4_scaled_pair_bf16(std::uint8_t code, std::uint8_t scale) {
#if __CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)
    // PTX 9.2 exposes native pair widening. E2M1 times a finite E4M3 scale needs
    // at most six significand bits, so the BF16 multiplication is exact.
    unsigned values, multiplier, result;
    asm("{\n"
        ".reg .b8 packed;\n"
        "mov.b32 {packed, _, _, _}, %1;\n"
        "cvt.rn.bf16x2.e2m1x2 %0, packed;\n"
        "}\n"
        : "=r"(values)
        : "r"(static_cast<unsigned>(code)));
    const auto scale_pair = static_cast<std::uint16_t>(scale | (static_cast<unsigned>(scale) << 8));
    multiplier            = fp8_e4m3x2_to_bf16x2_bits(scale_pair);
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(result) : "r"(values), "r"(multiplier));
    return result;
#else
    // CUDA 13.1 retains the exact FP32 expansion.
    const float2 values    = decode_nvfp4_e2m1x2(code);
    const float multiplier = decode_e4m3(scale);

    union {
        __nv_bfloat162 pair;
        unsigned bits;
    } result;

    result.pair = __floats2bfloat162_rn(values.x * multiplier, values.y * multiplier);
    return result.bits;
#endif
}

struct alignas(8) Nvfp4QuantizedK16 {
    std::uint32_t codes_lo;
    std::uint32_t codes_hi;
    std::uint8_t scale;
};

static_assert(alignof(Nvfp4QuantizedK16) == 8);

__device__ __forceinline__ void
pack_nvfp4_e2m1x16(const float2 (&values)[8], std::uint32_t& codes_lo, std::uint32_t& codes_hi) {
    asm volatile("{\n"
                 ".reg .b8 b0;\n"
                 ".reg .b8 b1;\n"
                 ".reg .b8 b2;\n"
                 ".reg .b8 b3;\n"
                 ".reg .b8 b4;\n"
                 ".reg .b8 b5;\n"
                 ".reg .b8 b6;\n"
                 ".reg .b8 b7;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b0, %3, %2;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b1, %5, %4;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b2, %7, %6;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b3, %9, %8;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b4, %11, %10;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b5, %13, %12;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b6, %15, %14;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b7, %17, %16;\n"
                 "mov.b32 %0, {b0,b1,b2,b3};\n"
                 "mov.b32 %1, {b4,b5,b6,b7};\n"
                 "}\n"
                 : "=r"(codes_lo), "=r"(codes_hi)
                 : "f"(values[0].x), "f"(values[0].y), "f"(values[1].x), "f"(values[1].y),
                   "f"(values[2].x), "f"(values[2].y), "f"(values[3].x), "f"(values[3].y),
                   "f"(values[4].x), "f"(values[4].y), "f"(values[5].x), "f"(values[5].y),
                   "f"(values[6].x), "f"(values[6].y), "f"(values[7].x), "f"(values[7].y));
}

// Calibrated activation profile, shared by BF16 inputs and FP32 MMA epilogues.
__device__ __forceinline__ std::uint8_t nvfp4_activation_scale(float maximum, float divisor) {
    return __nv_cvt_float_to_fp8(__fdiv_rn(divisor * maximum, 6.0F), __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ float2 nvfp4_activation_values(float2 values, float divisor,
                                                          float decoded_scale) {
    return make_float2(__fdiv_rn(values.x * divisor, decoded_scale),
                       __fdiv_rn(values.y * divisor, decoded_scale));
}

__device__ __forceinline__ std::uint8_t quantize_nvfp4_activation_pair(float2 values, float divisor,
                                                                       std::uint8_t scale) {
    if (scale == 0) { return 0; }
    return __nv_cvt_float2_to_fp4x2(nvfp4_activation_values(values, divisor, decode_e4m3(scale)),
                                    __NV_E2M1, cudaRoundNearest);
}

// BF16 input, group size 16, explicit tensor divisor. Codes use the rounded E4M3
// group scale; a scale that rounds to zero produces zero codes.
__device__ __forceinline__ Nvfp4QuantizedK16 quantize_nvfp4_k16(const __nv_bfloat16* source,
                                                                float input_scale_divisor) {
    const uint4 packed0                = load_vec<uint4>(source);
    const uint4 packed1                = load_vec<uint4>(source + 8);
    const std::uint32_t represented[8] = {
        packed0.x, packed0.y, packed0.z, packed0.w, packed1.x, packed1.y, packed1.z, packed1.w,
    };

    float2 values[8];
    float max_abs = 0.0F;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair] = bf16x2_bits_to_float2(represented[pair]);
        max_abs      = fmaxf(max_abs, fabsf(values[pair].x));
        max_abs      = fmaxf(max_abs, fabsf(values[pair].y));
    }

    Nvfp4QuantizedK16 result{};
    result.scale = nvfp4_activation_scale(max_abs, input_scale_divisor);
    if (result.scale == 0) { return result; }

    const float decoded_scale = decode_e4m3(result.scale);
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair] = nvfp4_activation_values(values[pair], input_scale_divisor, decoded_scale);
    }
    pack_nvfp4_e2m1x16(values, result.codes_lo, result.codes_hi);
    return result;
}

// FP32 producer input with the same explicit calibrated tensor divisor and rounded G16 scale.
__device__ __forceinline__ Nvfp4QuantizedK16 quantize_nvfp4_k16(const float* source,
                                                                float divisor) {
    float2 values[8];
    float maximum = 0.0f;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair] = load_vec<float2>(source + pair * 2);
        maximum      = fmaxf(maximum, fmaxf(fabsf(values[pair].x), fabsf(values[pair].y)));
    }
    Nvfp4QuantizedK16 result{};
    result.scale = nvfp4_activation_scale(maximum, divisor);
    if (!result.scale) return result;
    const float scale = decode_e4m3(result.scale);
#pragma unroll
    for (int pair = 0; pair < 8; ++pair)
        values[pair] = nvfp4_activation_values(values[pair], divisor, scale);
    pack_nvfp4_e2m1x16(values, result.codes_lo, result.codes_hi);
    return result;
}

// FP32 input, no tensor divisor. For a nonzero group, clamp its E4M3 scale to
// [2^-9, 448] before encoding. This is distinct from the BF16 activation profile.
struct Nvfp4Group16 {
    std::uint32_t codes_lo = 0;
    std::uint32_t codes_hi = 0;
    std::uint8_t scale     = 0;
};

__device__ __forceinline__ Nvfp4Group16 quantize_nvfp4_group16_f32(const float* source) {
    float2 values[8];
    float max_abs = 0.0F;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair] = make_float2(source[2 * pair], source[2 * pair + 1]);
        max_abs      = fmaxf(max_abs, fabsf(values[pair].x));
        max_abs      = fmaxf(max_abs, fabsf(values[pair].y));
    }

    Nvfp4Group16 result{};
    if (max_abs == 0.0F) return result;

    const float raw_scale         = __fdiv_rn(max_abs, 6.0F);
    const float bounded           = fminf(448.0F, fmaxf(0x1p-9F, raw_scale));
    result.scale                  = __nv_cvt_float_to_fp8(bounded, __NV_SATFINITE, __NV_E4M3);
    const float represented_scale = decode_e4m3(result.scale);
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair].x = __fdiv_rn(values[pair].x, represented_scale);
        values[pair].y = __fdiv_rn(values[pair].y, represented_scale);
    }
    pack_nvfp4_e2m1x16(values, result.codes_lo, result.codes_hi);
    return result;
}

__device__ __forceinline__ int4 nvfp4_dequant_f16x8(const std::uint8_t* codes,
                                                    std::uint8_t scale_code) {
    // Every finite E2M1 value times a legal nonnegative E4M3 cache scale is exactly representable
    // in FP16 (at most four product fraction bits and magnitude <= 2688). Half2 multiplication is
    // therefore the exact FP16 expansion boundary, not an additional approximation.
    const std::uint32_t packed = load_vec<std::uint32_t>(codes);
    const std::uint8_t* bytes  = reinterpret_cast<const std::uint8_t*>(&packed);
    __nv_fp8_e4m3 encoded_scale;
    encoded_scale.__x    = scale_code;
    const __half scale   = static_cast<__half>(encoded_scale);
    const __half2 scale2 = __halves2half2(scale, scale);
    unsigned half_bits[4];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        __nv_fp4x2_e2m1 encoded;
        encoded.__x         = bytes[pair];
        const __half2 value = __hmul2(static_cast<__half2>(encoded), scale2);
        half_bits[pair]     = *reinterpret_cast<const unsigned*>(&value);
    }
    return make_int4(static_cast<int>(half_bits[0]), static_cast<int>(half_bits[1]),
                     static_cast<int>(half_bits[2]), static_cast<int>(half_bits[3]));
}

struct Nvfp4F16x16 {
    int4 lo;
    int4 hi;
};

__device__ __forceinline__ Nvfp4F16x16 nvfp4_dequant_f16x16(const std::uint8_t* codes,
                                                            std::uint8_t scale_code) {
    const int2 packed         = load_vec<int2>(codes);
    const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(&packed);
    __nv_fp8_e4m3 encoded_scale;
    encoded_scale.__x    = scale_code;
    const __half scale   = static_cast<__half>(encoded_scale);
    const __half2 scale2 = __halves2half2(scale, scale);
    unsigned half_bits[8];
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        __nv_fp4x2_e2m1 encoded;
        encoded.__x         = bytes[pair];
        const __half2 value = __hmul2(static_cast<__half2>(encoded), scale2);
        half_bits[pair]     = *reinterpret_cast<const unsigned*>(&value);
    }
    return {
        make_int4(static_cast<int>(half_bits[0]), static_cast<int>(half_bits[1]),
                  static_cast<int>(half_bits[2]), static_cast<int>(half_bits[3])),
        make_int4(static_cast<int>(half_bits[4]), static_cast<int>(half_bits[5]),
                  static_cast<int>(half_bits[6]), static_cast<int>(half_bits[7])),
    };
}

} // namespace ninfer::ops::common
