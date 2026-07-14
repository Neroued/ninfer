#pragma once

#include "kernels/common/math.h"

// ninfer::kernels - embedding kernels. Dense copies BF16 rows; Q6 decodes
// ROW_SPLIT nibble, high, and scale planes.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::kernels {

inline constexpr std::int32_t kEmbedGatherQ6Group = 64;
inline constexpr std::int32_t kEmbedGatherQ6NibbleBpr = 32;
inline constexpr std::int32_t kEmbedGatherQ6HighBpr = 16;
inline constexpr std::int32_t kEmbedGatherQ6GroupsPerBlock = 2;

__device__ __forceinline__ int unpack_q6_code(const std::uint8_t* nibble,
                                              const std::uint8_t* high, int index) {
    const std::uint8_t low_byte = nibble[index >> 1];
    const std::uint32_t low = (index & 1) ? (low_byte >> 4) : (low_byte & 0x0fu);
    const int high_pos = index * 2;
    const std::uint32_t high_bits = (high[high_pos >> 3] >> (high_pos & 7)) & 0x03u;
    const std::uint32_t u = low | (high_bits << 4);
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

__global__ void embed_gather_q6_kernel(const std::int32_t* ids, const std::uint8_t* codes,
                                       const std::uint8_t* high,
                                       const std::uint8_t* scales,
                                       __nv_bfloat16* out, std::int32_t d, std::int32_t T,
                                       std::int32_t padded_d) {
    const std::int32_t kg         = padded_d / kEmbedGatherQ6Group;
    const std::int64_t n          = static_cast<std::int64_t>(d) * T;
    const std::int64_t start =
        blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;

    for (std::int64_t i = start; i < n; i += stride) {
        const std::int32_t t = static_cast<std::int32_t>(i / d);
        const std::int32_t k = static_cast<std::int32_t>(i - static_cast<std::int64_t>(t) * d);
        const std::int32_t row = ids[t];
        const std::int32_t g = k / kEmbedGatherQ6Group;
        const std::int32_t lane = k - g * kEmbedGatherQ6Group;
        const std::int64_t group_index = static_cast<std::int64_t>(row) * kg + g;
        const std::uint16_t scale_bits =
            static_cast<std::uint16_t>(scales[group_index * 2]) |
            static_cast<std::uint16_t>(
                static_cast<std::uint16_t>(scales[group_index * 2 + 1]) << 8);
        const float scale = __half2float(__ushort_as_half(scale_bits));
        const int code = unpack_q6_code(
            codes + group_index * kEmbedGatherQ6NibbleBpr,
            high + group_index * kEmbedGatherQ6HighBpr, lane);
        out[i] = __float2bfloat16(static_cast<float>(code) * scale);
    }
}

__launch_bounds__(kEmbedGatherQ6Group * kEmbedGatherQ6GroupsPerBlock)
__global__ void embed_gather_q6_grouped_kernel(const std::int32_t* ids,
                                               const std::uint8_t* codes,
                                               const std::uint8_t* high,
                                               const std::uint8_t* scales,
                                               __nv_bfloat16* out, std::int32_t d,
                                               std::int32_t T) {
    const std::int32_t kg = d / kEmbedGatherQ6Group;
    const std::int32_t group_blocks = div_up(kg, kEmbedGatherQ6GroupsPerBlock);
    const std::int32_t t = static_cast<std::int32_t>(blockIdx.x) / group_blocks;
    const std::int32_t block_group =
        static_cast<std::int32_t>(blockIdx.x) - t * group_blocks;
    const std::int32_t group_slot = threadIdx.x / kEmbedGatherQ6Group;
    const std::int32_t lane = threadIdx.x - group_slot * kEmbedGatherQ6Group;
    const std::int32_t g = block_group * kEmbedGatherQ6GroupsPerBlock + group_slot;
    if (g >= kg) { return; }

    const std::int32_t row = ids[t];
    const std::int64_t group_index = static_cast<std::int64_t>(row) * kg + g;

    int scale_bits = 0;
    if ((lane & 31) == 0) {
        scale_bits = static_cast<int>(scales[group_index * 2]) |
                     (static_cast<int>(scales[group_index * 2 + 1]) << 8);
    }
    scale_bits = __shfl_sync(0xffffffffu, scale_bits, 0);
    const float scale = __half2float(__ushort_as_half(static_cast<std::uint16_t>(scale_bits)));
    const int code = unpack_q6_code(codes + group_index * kEmbedGatherQ6NibbleBpr,
                                    high + group_index * kEmbedGatherQ6HighBpr, lane);
    const std::int64_t out_idx =
        static_cast<std::int64_t>(t) * d + static_cast<std::int64_t>(g) * kEmbedGatherQ6Group +
        lane;
    out[out_idx] = __float2bfloat16(static_cast<float>(code) * scale);
}

} // namespace ninfer::kernels
