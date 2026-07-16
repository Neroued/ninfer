#pragma once

#include "ops/common/math.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

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

} // namespace ninfer::ops::detail
