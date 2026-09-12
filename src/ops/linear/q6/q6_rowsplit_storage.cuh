#pragma once

#include "ops/common/math.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

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

struct Q6SimtDecodeAtom {
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
            const int q1               = ((static_cast<int>(byte >> 4) |
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

} // namespace ninfer::ops::detail
