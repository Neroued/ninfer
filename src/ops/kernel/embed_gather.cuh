#pragma once

#include "ops/common/math.h"

// ninfer::ops - embedding kernels. Dense copies BF16 rows; Q6 decodes
// ROW_SPLIT nibble, high, and scale planes.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr std::int32_t kEmbedGatherQ6Group          = 64;
inline constexpr std::int32_t kEmbedGatherQ6NibbleBpr      = 32;
inline constexpr std::int32_t kEmbedGatherQ6HighBpr        = 16;
inline constexpr std::int32_t kEmbedGatherQ6GroupsPerBlock = 2;
inline constexpr std::int32_t kEmbedGatherW8Group          = 32;
inline constexpr std::int32_t kEmbedGatherW8D              = 2048;
inline constexpr std::int32_t kEmbedGatherW8Groups         = kEmbedGatherW8D / kEmbedGatherW8Group;

__device__ __forceinline__ int unpack_q6_code(const std::uint8_t* nibble, const std::uint8_t* high,
                                              int index) {
    const std::uint8_t low_byte   = nibble[index >> 1];
    const std::uint32_t low       = (index & 1) ? (low_byte >> 4) : (low_byte & 0x0fu);
    const int high_pos            = index * 2;
    const std::uint32_t high_bits = (high[high_pos >> 3] >> (high_pos & 7)) & 0x03u;
    const std::uint32_t u         = low | (high_bits << 4);
    return (u & 0x20u) ? static_cast<int>(u) - 64 : static_cast<int>(u);
}

__global__ void embed_gather_dense_kernel(const std::int32_t* ids, const __nv_bfloat16* table,
                                          __nv_bfloat16* out, std::int32_t d, std::int32_t T) {
    const std::int64_t n      = static_cast<std::int64_t>(d) * T;
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const std::int32_t t = static_cast<std::int32_t>(i / d);
        const std::int32_t k = static_cast<std::int32_t>(i - static_cast<std::int64_t>(t) * d);
        out[i]               = table[static_cast<std::int64_t>(ids[t]) * d + k];
    }
}

__global__ void embed_gather_q6_kernel(const std::int32_t* ids, const std::uint8_t* codes,
                                       const std::uint8_t* high, const std::uint8_t* scales,
                                       __nv_bfloat16* out, std::int32_t d, std::int32_t T,
                                       std::int32_t padded_d) {
    const std::int32_t kg     = padded_d / kEmbedGatherQ6Group;
    const std::int64_t n      = static_cast<std::int64_t>(d) * T;
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;

    for (std::int64_t i = start; i < n; i += stride) {
        const std::int32_t t    = static_cast<std::int32_t>(i / d);
        const std::int32_t k    = static_cast<std::int32_t>(i - static_cast<std::int64_t>(t) * d);
        const std::int32_t row  = ids[t];
        const std::int32_t g    = k / kEmbedGatherQ6Group;
        const std::int32_t lane = k - g * kEmbedGatherQ6Group;
        const std::int64_t group_index = static_cast<std::int64_t>(row) * kg + g;
        const std::uint16_t scale_bits =
            static_cast<std::uint16_t>(scales[group_index * 2]) |
            static_cast<std::uint16_t>(static_cast<std::uint16_t>(scales[group_index * 2 + 1])
                                       << 8);
        const float scale = __half2float(__ushort_as_half(scale_bits));
        const int code    = unpack_q6_code(codes + group_index * kEmbedGatherQ6NibbleBpr,
                                           high + group_index * kEmbedGatherQ6HighBpr, lane);
        out[i]            = __float2bfloat16(static_cast<float>(code) * scale);
    }
}

__launch_bounds__(kEmbedGatherQ6Group* kEmbedGatherQ6GroupsPerBlock) __global__
    void embed_gather_q6_grouped_kernel(const std::int32_t* ids, const std::uint8_t* codes,
                                        const std::uint8_t* high, const std::uint8_t* scales,
                                        __nv_bfloat16* out, std::int32_t d, std::int32_t T) {
    const std::int32_t kg           = d / kEmbedGatherQ6Group;
    const std::int32_t group_blocks = div_up(kg, kEmbedGatherQ6GroupsPerBlock);
    const std::int32_t t            = static_cast<std::int32_t>(blockIdx.x) / group_blocks;
    const std::int32_t block_group  = static_cast<std::int32_t>(blockIdx.x) - t * group_blocks;
    const std::int32_t group_slot   = threadIdx.x / kEmbedGatherQ6Group;
    const std::int32_t lane         = threadIdx.x - group_slot * kEmbedGatherQ6Group;
    const std::int32_t g            = block_group * kEmbedGatherQ6GroupsPerBlock + group_slot;
    if (g >= kg) { return; }

    const std::int32_t row         = ids[t];
    const std::int64_t group_index = static_cast<std::int64_t>(row) * kg + g;

    int scale_bits = 0;
    if ((lane & 31) == 0) {
        scale_bits = static_cast<int>(scales[group_index * 2]) |
                     (static_cast<int>(scales[group_index * 2 + 1]) << 8);
    }
    scale_bits        = __shfl_sync(0xffffffffu, scale_bits, 0);
    const float scale = __half2float(__ushort_as_half(static_cast<std::uint16_t>(scale_bits)));
    const int code    = unpack_q6_code(codes + group_index * kEmbedGatherQ6NibbleBpr,
                                       high + group_index * kEmbedGatherQ6HighBpr, lane);
    const std::int64_t out_idx = static_cast<std::int64_t>(t) * d +
                                 static_cast<std::int64_t>(g) * kEmbedGatherQ6Group + lane;
    out[out_idx] = __float2bfloat16(static_cast<float>(code) * scale);
}

__global__ void embed_gather_w8_kernel(const std::int32_t* ids, const std::uint8_t* codes,
                                       const std::uint8_t* scales, __nv_bfloat16* out,
                                       std::int32_t d, std::int32_t T, std::int32_t padded_d) {
    const std::int32_t kg     = padded_d / kEmbedGatherW8Group;
    const std::int64_t n      = static_cast<std::int64_t>(d) * T;
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const std::int32_t t    = static_cast<std::int32_t>(i / d);
        const std::int32_t k    = static_cast<std::int32_t>(i - static_cast<std::int64_t>(t) * d);
        const std::int32_t row  = ids[t];
        const std::int32_t g    = k / kEmbedGatherW8Group;
        const std::int32_t lane = k - g * kEmbedGatherW8Group;
        const std::int64_t group_index = static_cast<std::int64_t>(row) * kg + g;
        const std::uint16_t scale_bits =
            static_cast<std::uint16_t>(scales[group_index * 2]) |
            static_cast<std::uint16_t>(static_cast<std::uint16_t>(scales[group_index * 2 + 1])
                                       << 8);
        const float scale = __half2float(__ushort_as_half(scale_bits));
        const auto code = static_cast<std::int8_t>(codes[group_index * kEmbedGatherW8Group + lane]);
        out[i]          = __float2bfloat16(static_cast<float>(code) * scale);
    }
}

__launch_bounds__(32) __global__
    void embed_gather_w8_grouped_2048_kernel(const std::int32_t* ids, const std::uint8_t* codes,
                                             const std::uint8_t* scales, __nv_bfloat16* out) {
    const std::int32_t t   = static_cast<std::int32_t>(blockIdx.x) / kEmbedGatherW8Groups;
    const std::int32_t g   = static_cast<std::int32_t>(blockIdx.x) - t * kEmbedGatherW8Groups;
    const std::int32_t row = ids[t];
    const std::int64_t group_index = static_cast<std::int64_t>(row) * kEmbedGatherW8Groups + g;

    int scale_bits = 0;
    if (threadIdx.x == 0) {
        scale_bits = static_cast<int>(scales[group_index * 2]) |
                     (static_cast<int>(scales[group_index * 2 + 1]) << 8);
    }
    scale_bits        = __shfl_sync(0xffffffffu, scale_bits, 0);
    const float scale = __half2float(__ushort_as_half(static_cast<std::uint16_t>(scale_bits)));
    const auto code   = static_cast<std::int8_t>(
        codes[group_index * kEmbedGatherW8Group + static_cast<std::int32_t>(threadIdx.x)]);
    const std::int64_t out_idx = static_cast<std::int64_t>(t) * kEmbedGatherW8D +
                                 static_cast<std::int64_t>(g) * kEmbedGatherW8Group + threadIdx.x;
    out[out_idx] = __float2bfloat16(static_cast<float>(code) * scale);
}

__launch_bounds__(256) __global__
    void embed_gather_w8_row_2048_kernel(const std::int32_t* ids, const std::uint8_t* codes,
                                         const std::uint8_t* scales, __nv_bfloat16* out) {
    const int tid = static_cast<int>(threadIdx.x);
    const int t   = static_cast<int>(blockIdx.x);
    const int row = ids[t];

    constexpr int kValuesPerThread = kEmbedGatherW8D / 256;
    static_assert(kValuesPerThread == 8);
    const int k = tid * kValuesPerThread;
    const std::int64_t group_index =
        static_cast<std::int64_t>(row) * kEmbedGatherW8Groups + k / kEmbedGatherW8Group;
    const std::uint16_t scale_bits =
        static_cast<std::uint16_t>(scales[group_index * 2]) |
        static_cast<std::uint16_t>(static_cast<std::uint16_t>(scales[group_index * 2 + 1]) << 8);
    const float scale    = __half2float(__ushort_as_half(scale_bits));
    const auto* code_row = codes + static_cast<std::int64_t>(row) * kEmbedGatherW8D;
    const uint2 packed   = *reinterpret_cast<const uint2*>(code_row + k);
    auto* out_pairs =
        reinterpret_cast<__nv_bfloat162*>(out + static_cast<std::int64_t>(t) * kEmbedGatherW8D + k);
#pragma unroll
    for (int word_index = 0; word_index < 2; ++word_index) {
        const std::uint32_t word = word_index == 0 ? packed.x : packed.y;
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
            const int shift = pair * 16;
            const auto q0   = static_cast<std::int8_t>((word >> shift) & 0xffu);
            const auto q1   = static_cast<std::int8_t>((word >> (shift + 8)) & 0xffu);
            out_pairs[word_index * 2 + pair] = __floats2bfloat162_rn(
                static_cast<float>(q0) * scale, static_cast<float>(q1) * scale);
        }
    }
}

} // namespace ninfer::ops
