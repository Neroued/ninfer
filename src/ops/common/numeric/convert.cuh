#pragma once

#include "ops/common/cuda/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::common {

template <int Width>
struct alignas(2 * Width) Bf16Vector {
    __nv_bfloat16 word[Width];

    __device__ static Bf16Vector load(const __nv_bfloat16* p) {
        return *reinterpret_cast<const Bf16Vector*>(p);
    }

    __device__ void store(__nv_bfloat16* p) const { *reinterpret_cast<Bf16Vector*>(p) = *this; }
};

template <int Pairs>
struct alignas(Pairs* static_cast<int>(sizeof(__nv_bfloat162))) Bf16PairPack {
    static_assert(Pairs == 1 || Pairs == 2 || Pairs == 4);
    __nv_bfloat162 pair[Pairs];
};

using Bf16x4Pack = Bf16PairPack<2>;
using Bf16x8Pack = Bf16PairPack<4>;

union alignas(16) Bf16x8Bits {
    uint4 raw;
    __nv_bfloat162 pair[4];
};

static_assert(sizeof(Bf16x8Bits) == 16);

static_assert(sizeof(Bf16x4Pack) == 8);
static_assert(sizeof(Bf16x8Pack) == 16);

// Exact widening of the two signed bytes in the low 16 bits.
__device__ __forceinline__ unsigned int8x2_to_bf16x2_bits(unsigned values) {
    union PairBits {
        __nv_bfloat162 pair;
        unsigned bits;
    };

    PairBits biased;
    biased.bits          = __byte_perm(values, 0x43004300u, 0x7150) & 0xff7fff7fu;
    const unsigned signs = (values & 0x80u) | ((values & 0x8000u) << 8);
    PairBits bias;
    bias.bits = 0x43004300u | signs;
    PairBits result;
    result.pair = __hsub2_rn(biased.pair, bias.pair);
    return result.bits;
}

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

__device__ __forceinline__ float load_fp16_unaligned(const std::uint8_t* source) {
    const auto bits =
        static_cast<std::uint16_t>(source[0]) | (static_cast<std::uint16_t>(source[1]) << 8);
    return __half2float(__ushort_as_half(static_cast<std::uint16_t>(bits)));
}

} // namespace ninfer::ops::common
