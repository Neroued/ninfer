#pragma once

#include "ops/common/numeric/convert.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::common {

struct Q4RowSplitStorage {
    static constexpr int kGroupK             = 64;
    static constexpr int kCodeBytesPerGroup  = 32;
    static constexpr int kScaleBytesPerGroup = 2;
};

struct Q4SimtDecodeAtom {
    __device__ static __forceinline__ void
    decode_eight(std::uint32_t packed, std::uint16_t scale_bits, float (&weights)[8]) {
        const std::uint32_t word = packed ^ 0x88888888u;
        const float scale        = __half2float(__ushort_as_half(scale_bits));
        const __half2 bias       = __half2half2(__ushort_as_half(0x6408)); // 1032.0
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const std::uint32_t bits = ((word >> (4 * pair)) & 0x000f000fu) | 0x64006400u;
            const __half2 decoded    = __hsub2(half2_from_bits(bits), bias);
            const float2 values      = __half22float2(decoded);
            weights[pair]            = values.x * scale;
            weights[pair + 4]        = values.y * scale;
        }
    }

    __device__ static __forceinline__ void
    decode_pair(std::uint8_t packed, std::uint16_t scale_bits, float& w0, float& w1) {
        const float scale = __half2float(__ushort_as_half(scale_bits));
        const int q0      = (static_cast<int>(packed & 0x0fu) ^ 0x08) - 0x08;
        const int q1      = (static_cast<int>(packed >> 4) ^ 0x08) - 0x08;
        w0                = static_cast<float>(q0) * scale;
        w1                = static_cast<float>(q1) * scale;
    }
};

struct Q4MmaDecodeAtom {
    // Eight packed codes -> four bf16 pairs, in weight order; out[i] holds the pair for
    // lane = 4 * chunk + i. A nibble n decodes to the integer (n ^ 8) - 8, which lies in [-8, 7]
    // and which bf16 represents exactly, so forming 128 + (n ^ 8) and subtracting 136 lands on it.
    // The values are unscaled: callers of this overload fold the group scale into the
    // accumulation, unlike decode_pair below.
    static __device__ __forceinline__ void decode_eight(unsigned word, unsigned (&out)[4]) {
        const unsigned kBias  = 0x43084308u; // bf16 136.0 in both halves
        const unsigned kMagic = 0x43004300u; // bf16 128.0 in both halves
        word ^= 0x88888888u;
        const unsigned lo        = word & 0x0f0f0f0fu;
        const unsigned hi        = (word >> 4) & 0x0f0f0f0fu;
        const unsigned even      = __byte_perm(lo, hi, 0x5140);
        const unsigned odd       = __byte_perm(lo, hi, 0x7362);
        const unsigned biased[4] = {
            __byte_perm(even, kMagic, 0x7150), __byte_perm(even, kMagic, 0x7372),
            __byte_perm(odd, kMagic, 0x7150), __byte_perm(odd, kMagic, 0x7372)};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const __nv_bfloat162 value =
                __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&biased[i]),
                        *reinterpret_cast<const __nv_bfloat162*>(&kBias));
            out[i] = *reinterpret_cast<const unsigned*>(&value);
        }
    }

    static __device__ __forceinline__ __nv_bfloat162 decode_pair(const std::uint8_t* codes,
                                                                 const std::uint8_t* scale_ptr,
                                                                 std::int64_t group_index,
                                                                 int lane) {
        const float scale =
            __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scale_ptr)));
        const std::uint8_t packed =
            codes[group_index * Q4RowSplitStorage::kCodeBytesPerGroup + lane];
        const int q0 = (static_cast<int>(packed & 0x0fu) ^ 0x08) - 0x08;
        const int q1 = (static_cast<int>(packed >> 4) ^ 0x08) - 0x08;
        return __floats2bfloat162_rn(static_cast<float>(q0) * scale,
                                     static_cast<float>(q1) * scale);
    }
};

struct Q5RowSplitStorage {
    static constexpr int kGroupK             = 64;
    static constexpr int kCodeBytesPerGroup  = 32;
    static constexpr int kHighBytesPerGroup  = 8;
    static constexpr int kScaleBytesPerGroup = 2;

    // A chunk is the eight codes one thread decodes, so it spans four packed bytes.
    static constexpr int kCodeBytesPerChunk = 4;
    static constexpr int kHighBytesPerChunk =
        kHighBytesPerGroup / (kCodeBytesPerGroup / kCodeBytesPerChunk);
    static_assert(kHighBytesPerChunk * (kCodeBytesPerGroup / kCodeBytesPerChunk) ==
                      kHighBytesPerGroup,
                  "the high-bit plane must divide evenly across a group's chunks");
};

struct Q5ScalarDecodeAtom {
    static constexpr int kGroupK = Q5RowSplitStorage::kGroupK;

    __device__ static __forceinline__ void
    load_pair(const std::uint8_t* codes, const std::uint8_t* high, const std::uint8_t* scales,
              std::int64_t group_index, int lane, float& w0, float& w1) {
        decode_pair(codes, high, scales, group_index, lane, w0, w1);
    }

    __device__ static __forceinline__ void
    decode_pair(const std::uint8_t* codes, const std::uint8_t* high, const std::uint8_t* scales,
                std::int64_t group_index, int lane, float& w0, float& w1) {
        const float scale = __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
            scales + group_index * Q5RowSplitStorage::kScaleBytesPerGroup)));
        const std::uint8_t packed =
            codes[group_index * Q5RowSplitStorage::kCodeBytesPerGroup + lane];
        const std::uint8_t high_byte =
            high[group_index * Q5RowSplitStorage::kHighBytesPerGroup + (lane >> 2)];
        const int shift = (lane & 3) * 2;
        const int q0 =
            ((static_cast<int>(packed & 0x0fu) | (((high_byte >> shift) & 1) << 4)) ^ 0x10) - 0x10;
        const int q1 =
            ((static_cast<int>(packed >> 4) | (((high_byte >> (shift + 1)) & 1) << 4)) ^ 0x10) -
            0x10;
        w0 = static_cast<float>(q0) * scale;
        w1 = static_cast<float>(q1) * scale;
    }
};

struct Q5SimtDecodeAtom {
    __device__ static __forceinline__ void
    load_eight(const std::uint8_t* codes, const std::uint8_t* high, const std::uint8_t* scales,
               std::int64_t group_index, int lane_in_group, float (&weights)[8]) {
        const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
            codes + group_index * Q5RowSplitStorage::kCodeBytesPerGroup + lane_in_group * 4);
        const std::uint8_t high_bits =
            high[group_index * Q5RowSplitStorage::kHighBytesPerGroup + lane_in_group];
        const auto scale_bits = *reinterpret_cast<const std::uint16_t*>(
            scales + group_index * Q5RowSplitStorage::kScaleBytesPerGroup);
        decode_eight(packed, high_bits, scale_bits, weights);
    }

    __device__ static __forceinline__ void decode_eight(std::uint32_t packed,
                                                        std::uint8_t high_bits,
                                                        std::uint16_t scale_bits,
                                                        float (&weights)[8]) {
        const std::uint32_t high = static_cast<std::uint32_t>(high_bits) ^ 0xffu;
        const float scale        = __half2float(__ushort_as_half(scale_bits));
        const __half2 bias       = __half2half2(__ushort_as_half(0x6410)); // 1040.0
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            std::uint32_t bits = ((packed >> (4 * pair)) & 0x000f000fu) | 0x64006400u;
            bits |= (((high >> pair) & 1u) << 4) | (((high >> (pair + 4)) & 1u) << 20);
            const __half2 decoded = __hsub2(half2_from_bits(bits), bias);
            const float2 values   = __half22float2(decoded);
            weights[pair]         = values.x * scale;
            weights[pair + 4]     = values.y * scale;
        }
    }
};

struct Q5MmaDecodeAtom {
    // Four packed bytes plus its high-bit byte -> four bf16 pairs, in weight order; out[i] holds
    // the pair decode_pair produces at lane = 4 * chunk + i.
    static __device__ __forceinline__ void
    decode_eight(unsigned word, const std::uint8_t* high_chunk, float scale, unsigned (&out)[4]) {
        const unsigned high = high_chunk[0];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned byte = (word >> (8 * i)) & 0xffu;
            const int q0        = ((static_cast<int>(byte & 0x0fu) |
                             static_cast<int>(((high >> (2 * i)) & 1u) << 4)) ^
                            0x10) -
                           0x10;
            const int q1 = ((static_cast<int>(byte >> 4) |
                             static_cast<int>(((high >> (2 * i + 1)) & 1u) << 4)) ^
                            0x10) -
                           0x10;
            const __nv_bfloat162 value = __floats2bfloat162_rn(static_cast<float>(q0) * scale,
                                                               static_cast<float>(q1) * scale);
            out[i]                     = *reinterpret_cast<const unsigned*>(&value);
        }
    }

    static __device__ __forceinline__ __nv_bfloat162 decode_pair(const std::uint8_t* staged_codes,
                                                                 const std::uint8_t* staged_high,
                                                                 const std::uint8_t* scale_ptr,
                                                                 std::int64_t staged_group_index,
                                                                 int lane) {
        const float scale =
            __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scale_ptr)));
        const std::uint8_t packed =
            staged_codes[staged_group_index * Q5RowSplitStorage::kCodeBytesPerGroup + lane];
        const std::uint8_t high_byte =
            staged_high[staged_group_index * Q5RowSplitStorage::kHighBytesPerGroup + (lane >> 2)];
        const int shift = (lane & 3) * 2;
        const int q0 =
            ((static_cast<int>(packed & 0x0fu) | (((high_byte >> shift) & 1) << 4)) ^ 0x10) - 0x10;
        const int q1 =
            ((static_cast<int>(packed >> 4) | (((high_byte >> (shift + 1)) & 1) << 4)) ^ 0x10) -
            0x10;
        return __floats2bfloat162_rn(static_cast<float>(q0) * scale,
                                     static_cast<float>(q1) * scale);
    }
};

struct Q6RowSplitStorage {
    static constexpr int kGroupK             = 64;
    static constexpr int kCodeBytesPerGroup  = 32;
    static constexpr int kHighBytesPerGroup  = 16;
    static constexpr int kScaleBytesPerGroup = 2;

    // A chunk is the eight codes one thread decodes, so it spans four packed bytes.
    static constexpr int kCodeBytesPerChunk = 4;
    static constexpr int kHighBytesPerChunk =
        kHighBytesPerGroup / (kCodeBytesPerGroup / kCodeBytesPerChunk);
    static_assert(kHighBytesPerChunk * (kCodeBytesPerGroup / kCodeBytesPerChunk) ==
                      kHighBytesPerGroup,
                  "the high-bit plane must divide evenly across a group's chunks");
};

__device__ __forceinline__ int unpack_q6_code(const std::uint8_t* codes, const std::uint8_t* high,
                                              int index) {
    const std::uint8_t low_byte   = codes[index >> 1];
    const std::uint32_t low       = (index & 1) ? (low_byte >> 4) : (low_byte & 0x0fu);
    const int high_pos            = index * 2;
    const std::uint32_t high_bits = (high[high_pos >> 3] >> (high_pos & 7)) & 0x03u;
    const std::uint32_t value     = low | (high_bits << 4);
    return (value & 0x20u) ? static_cast<int>(value) - 64 : static_cast<int>(value);
}

struct Q6SimtDecodeAtom {
    __device__ static __forceinline__ void
    load_eight(const std::uint8_t* codes, const std::uint8_t* high, const std::uint8_t* scales,
               std::int64_t group_index, int lane_in_group, float (&weights)[8]) {
        const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
            codes + group_index * Q6RowSplitStorage::kCodeBytesPerGroup + lane_in_group * 4);
        const std::uint16_t high_bits = *reinterpret_cast<const std::uint16_t*>(
            high + group_index * Q6RowSplitStorage::kHighBytesPerGroup + lane_in_group * 2);
        const auto scale_bits = *reinterpret_cast<const std::uint16_t*>(
            scales + group_index * Q6RowSplitStorage::kScaleBytesPerGroup);
        decode_eight(packed, high_bits, scale_bits, weights);
    }

    __device__ static __forceinline__ void decode_eight(std::uint32_t packed,
                                                        std::uint16_t high_bits,
                                                        std::uint16_t scale_bits,
                                                        float (&weights)[8]) {
        const std::uint32_t high = static_cast<std::uint32_t>(high_bits) ^ 0xaaaau;
        const float scale        = __half2float(__ushort_as_half(scale_bits));
        const __half2 bias       = __half2half2(__ushort_as_half(0x6420)); // 1056.0
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            std::uint32_t bits = ((packed >> (4 * pair)) & 0x000f000fu) | 0x64006400u;
            bits |= (((high >> (2 * pair)) & 3u) << 4) | (((high >> (2 * pair + 8)) & 3u) << 20);
            const __half2 decoded = __hsub2(half2_from_bits(bits), bias);
            const float2 values   = __half22float2(decoded);
            weights[pair]         = values.x * scale;
            weights[pair + 4]     = values.y * scale;
        }
    }
};

struct Q6MmaDecodeAtom {
    // Four packed bytes plus their two high-bit bytes -> four bf16 pairs, in weight order; out[i]
    // holds the pair decode_pair produces at lane = 4 * chunk + i.
    static __device__ __forceinline__ void
    decode_eight(unsigned word, const std::uint8_t* high_chunk, float scale, unsigned (&out)[4]) {
        const unsigned high0 = high_chunk[0];
        const unsigned high1 = high_chunk[1];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned byte = (word >> (8 * i)) & 0xffu;
            const unsigned high = i < 2 ? high0 : high1;
            const int shift     = (i & 1) * 4;
            const int q0 =
                ((static_cast<int>(byte & 0x0fu) | static_cast<int>(((high >> shift) & 3u) << 4)) ^
                 0x20) -
                0x20;
            const int q1 = ((static_cast<int>(byte >> 4) |
                             static_cast<int>(((high >> (shift + 2)) & 3u) << 4)) ^
                            0x20) -
                           0x20;
            const __nv_bfloat162 value = __floats2bfloat162_rn(static_cast<float>(q0) * scale,
                                                               static_cast<float>(q1) * scale);
            out[i]                     = *reinterpret_cast<const unsigned*>(&value);
        }
    }

    static __device__ __forceinline__ __nv_bfloat162 decode_pair(const std::uint8_t* staged_codes,
                                                                 const std::uint8_t* staged_high,
                                                                 const std::uint8_t* scale_ptr,
                                                                 std::int64_t staged_group_index,
                                                                 int lane) {
        const float scale =
            __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scale_ptr)));
        const std::uint8_t packed =
            staged_codes[staged_group_index * Q6RowSplitStorage::kCodeBytesPerGroup + lane];
        const std::uint8_t high_byte =
            staged_high[staged_group_index * Q6RowSplitStorage::kHighBytesPerGroup + (lane >> 1)];
        const int shift = (lane & 1) * 4;
        const int q0 =
            ((static_cast<int>(packed & 0x0fu) | (((high_byte >> shift) & 3) << 4)) ^ 0x20) - 0x20;
        const int q1 =
            ((static_cast<int>(packed >> 4) | (((high_byte >> (shift + 2)) & 3) << 4)) ^ 0x20) -
            0x20;
        return __floats2bfloat162_rn(static_cast<float>(q0) * scale,
                                     static_cast<float>(q1) * scale);
    }
};

struct Q8RowSplitStorage {
    static constexpr int kGroupK             = 32;
    static constexpr int kCodeBytesPerGroup  = 32;
    static constexpr int kHighBytesPerGroup  = 0;
    static constexpr int kScaleBytesPerGroup = 2;
};

__device__ __forceinline__ __nv_bfloat162 dequant_q8_pair(std::uint16_t packed, float scale) {
    const auto lo = static_cast<std::int8_t>(packed & 0xffu);
    const auto hi = static_cast<std::int8_t>(packed >> 8);
    return __floats2bfloat162_rn(static_cast<float>(lo) * scale, static_cast<float>(hi) * scale);
}

struct Q8ScalarDecodeAtom {
    __device__ static __forceinline__ float load_one(const std::uint8_t* codes,
                                                     const std::uint8_t* scales,
                                                     std::int64_t group_index, int lane) {
        const float scale = __half2float(
            __ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scales + group_index * 2)));
        return static_cast<float>(static_cast<std::int8_t>(codes[group_index * kGroupK + lane])) *
               scale;
    }

    static constexpr int kGroupK = Q8RowSplitStorage::kGroupK;

    __device__ static __forceinline__ void
    load_pair(const std::uint8_t* codes, const std::uint8_t* /*high*/, const std::uint8_t* scales,
              std::int64_t group_index, int lane, float& w0, float& w1) {
        if (lane >= kGroupK / 2) {
            w0 = 0.0f;
            w1 = 0.0f;
            return;
        }
        const float scale = __half2float(
            __ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scales + group_index * 2)));
        const std::uint8_t* packed = codes + group_index * Q8RowSplitStorage::kCodeBytesPerGroup +
                                     static_cast<std::int64_t>(lane) * 2;
        w0 = static_cast<float>(static_cast<std::int8_t>(packed[0])) * scale;
        w1 = static_cast<float>(static_cast<std::int8_t>(packed[1])) * scale;
    }
};

struct Int8Fp16Scale {
    __half scale;
    float inverse_scale;
};

// Exact persistent group-scale boundary shared by standalone and fused append. The stored scale is
// FP16-RNE(absmax/127); codes always use the reciprocal of that represented FP16 value.
__device__ __forceinline__ Int8Fp16Scale int8_fp16_scale(float absmax) {
    const __half scale            = __float2half_rn(absmax > 0.0f ? absmax / 127.0f : 0.0f);
    const float represented_scale = __half2float(scale);
    return {
        .scale         = scale,
        .inverse_scale = represented_scale > 0.0f ? 1.0f / represented_scale : 0.0f,
    };
}

__device__ __forceinline__ std::int8_t int8_quant_code(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return static_cast<std::int8_t>(0); }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-127, min(127, q));
    return static_cast<std::int8_t>(q);
}

__device__ __forceinline__ int4 int8_dequant_bf16x8(const std::int8_t* codes8, float s) {
    const int2 raw       = load_vec<int2>(codes8);
    const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&raw);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = static_cast<float>(c[2 * i]) * s;
        const float x1 = static_cast<float>(c[2 * i + 1]) * s;
        packed[i]      = pack_bf16x2(x0, x1);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

// Eight code bytes share one represented FP16 scale; expansion multiplies in FP16.
__device__ __forceinline__ int4 int8_dequant_f16x8(const std::int8_t* codes8, __half scale) {
    const int2 raw       = load_vec<int2>(codes8);
    const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&raw);
    const __half2 s2     = __halves2half2(scale, scale);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2 code2 =
            __floats2half2_rn(static_cast<float>(c[2 * i]), static_cast<float>(c[2 * i + 1]));
        const __half2 value2 = __hmul2(code2, s2);
        packed[i]            = *reinterpret_cast<const unsigned*>(&value2);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

} // namespace ninfer::ops::common
