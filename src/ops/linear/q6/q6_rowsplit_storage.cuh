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
