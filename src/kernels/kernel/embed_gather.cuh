#pragma once

// qus::kernels - embed_gather kernels. Dense copies BF16 rows; Q6 decodes the
// RowGroupedG64 payload format: fp16 scale followed by 48 LSB-first Q6 bytes.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace qus::kernels {

__device__ __forceinline__ int unpack_q6_code(const std::uint8_t* packed, int index) {
    std::uint32_t u = 0;
    for (int bit = 0; bit < 6; ++bit) {
        const int pos = index * 6 + bit;
        u |= static_cast<std::uint32_t>((packed[pos >> 3] >> (pos & 7)) & 1u) << bit;
    }
    return (u & 0x20u) ? static_cast<int>(u) - 64 : static_cast<int>(u);
}

__global__ void embed_gather_dense_kernel(const std::int32_t* ids, const __nv_bfloat16* table,
                                          __nv_bfloat16* out, std::int32_t d, std::int32_t T) {
    const std::int64_t n = static_cast<std::int64_t>(d) * T;
    const std::int64_t start =
        blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const std::int32_t t = static_cast<std::int32_t>(i / d);
        const std::int32_t k = static_cast<std::int32_t>(i - static_cast<std::int64_t>(t) * d);
        out[i] = table[static_cast<std::int64_t>(ids[t]) * d + k];
    }
}

__global__ void embed_gather_q6_kernel(const std::int32_t* ids, const std::uint8_t* payload,
                                       __nv_bfloat16* out, std::int32_t d, std::int32_t T,
                                       std::int32_t padded_d) {
    constexpr std::int32_t kGroup = 64;
    constexpr std::int32_t kBpr   = 48;
    constexpr std::int32_t kRoww  = 2 + kBpr;
    const std::int32_t kg         = padded_d / kGroup;
    const std::int64_t n          = static_cast<std::int64_t>(d) * T;
    const std::int64_t start =
        blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;

    for (std::int64_t i = start; i < n; i += stride) {
        const std::int32_t t = static_cast<std::int32_t>(i / d);
        const std::int32_t k = static_cast<std::int32_t>(i - static_cast<std::int64_t>(t) * d);
        const std::int32_t row = ids[t];
        const std::int32_t g = k / kGroup;
        const std::int32_t lane = k - g * kGroup;
        const std::int64_t off =
            (static_cast<std::int64_t>(row) * kg + g) * static_cast<std::int64_t>(kRoww);
        const std::uint16_t scale_bits =
            static_cast<std::uint16_t>(payload[off]) |
            static_cast<std::uint16_t>(static_cast<std::uint16_t>(payload[off + 1]) << 8);
        const float scale = __half2float(__ushort_as_half(scale_bits));
        const int code = unpack_q6_code(payload + off + 2, lane);
        out[i] = __float2bfloat16(static_cast<float>(code) * scale);
    }
}

} // namespace qus::kernels
