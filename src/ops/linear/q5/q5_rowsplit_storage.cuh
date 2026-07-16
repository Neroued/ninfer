#pragma once

#include "ops/common/math.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Q5RowSplitStorage {
    static constexpr int kGroupK             = 64;
    static constexpr int kCodeBytesPerGroup  = 32;
    static constexpr int kHighBytesPerGroup  = 8;
    static constexpr int kScaleBytesPerGroup = 2;
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

} // namespace ninfer::ops::detail
