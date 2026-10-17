#pragma once

#include "ops/common/numeric/math.h"
#include "ops/common/quant/integer.cuh"
#include "ops/common/quant/fp8.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail::embedding {
inline constexpr std::int32_t kEmbedGatherQ6Group = common::Q6RowSplitStorage::kGroupK;
inline constexpr std::int32_t kEmbedGatherQ6NibbleBpr =
    common::Q6RowSplitStorage::kCodeBytesPerGroup;
inline constexpr std::int32_t kEmbedGatherQ6HighBpr = common::Q6RowSplitStorage::kHighBytesPerGroup;
inline constexpr std::int32_t kEmbedGatherQ6GroupsPerBlock = 2;
inline constexpr std::int32_t kEmbedGatherQ8Group          = common::Q8RowSplitStorage::kGroupK;
inline constexpr std::int32_t kEmbedGatherQ8D              = 2048;
inline constexpr std::int32_t kEmbedGatherQ8Groups         = kEmbedGatherQ8D / kEmbedGatherQ8Group;
inline constexpr std::int32_t kEmbedGatherFp8D             = 5120;

template <int BlocksPerToken, int Threads>
__launch_bounds__(Threads) __global__
    void embed_gather_fp8_kernel(const std::int32_t* ids, const std::uint8_t* codes,
                                 const __nv_bfloat16* scales, __nv_bfloat16* out) {
    static_assert(kEmbedGatherFp8D % BlocksPerToken == 0);
    constexpr int kValuesPerBlock = kEmbedGatherFp8D / BlocksPerToken;
    static_assert(kValuesPerBlock % 4 == 0);
    constexpr int kWordsPerBlock = kValuesPerBlock / 4;

    const int token   = static_cast<int>(blockIdx.x) / BlocksPerToken;
    const int split   = static_cast<int>(blockIdx.x) - token * BlocksPerToken;
    const int row     = ids[token];
    const float scale = __bfloat162float(scales[row]);

    const int split_offset = split * kValuesPerBlock;
    const auto* code_row   = codes + static_cast<std::int64_t>(row) * kEmbedGatherFp8D;
    auto* output_column    = out + static_cast<std::int64_t>(token) * kEmbedGatherFp8D;
    for (int word_index = static_cast<int>(threadIdx.x); word_index < kWordsPerBlock;
         word_index += Threads) {
        const int offset    = split_offset + word_index * 4;
        const unsigned word = *reinterpret_cast<const unsigned*>(code_row + offset);
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
            const float2 decoded =
                common::fp8_code2_to_float2(static_cast<std::uint16_t>(word >> (pair * 16)));
            reinterpret_cast<__nv_bfloat162*>(output_column + offset + pair * 2)[0] =
                __floats2bfloat162_rn(decoded.x * scale, decoded.y * scale);
        }
    }
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

template <int Blocks, int Threads>
__launch_bounds__(Threads) __global__
    void embed_gather_bf16_d2560_kernel(const std::int32_t* ids, const uint4* table,
                                        uint4* output) {
    constexpr int Vectors = 2560 / 8, Tile = Vectors / Blocks;
    static_assert(Vectors % Blocks == 0);
    const auto token = blockIdx.x / Blocks, split = blockIdx.x % Blocks;
    const auto row = std::int64_t(ids[token]);
    for (int i = threadIdx.x; i < Tile; i += Threads) {
        const auto vector                              = split * Tile + i;
        output[std::int64_t(token) * Vectors + vector] = table[row * Vectors + vector];
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
        const float scale              = common::load_fp16_unaligned(scales + group_index * 2);
        const int code = common::unpack_q6_code(codes + group_index * kEmbedGatherQ6NibbleBpr,
                                                high + group_index * kEmbedGatherQ6HighBpr, lane);
        out[i]         = __float2bfloat16(static_cast<float>(code) * scale);
    }
}

__launch_bounds__(kEmbedGatherQ6Group* kEmbedGatherQ6GroupsPerBlock) __global__
    void embed_gather_q6_grouped_kernel(const std::int32_t* ids, const std::uint8_t* codes,
                                        const std::uint8_t* high, const std::uint8_t* scales,
                                        __nv_bfloat16* out, std::int32_t d, std::int32_t T) {
    const std::int32_t kg           = d / kEmbedGatherQ6Group;
    const std::int32_t group_blocks = common::div_up(kg, kEmbedGatherQ6GroupsPerBlock);
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
    const int code    = common::unpack_q6_code(codes + group_index * kEmbedGatherQ6NibbleBpr,
                                               high + group_index * kEmbedGatherQ6HighBpr, lane);
    const std::int64_t out_idx = static_cast<std::int64_t>(t) * d +
                                 static_cast<std::int64_t>(g) * kEmbedGatherQ6Group + lane;
    out[out_idx] = __float2bfloat16(static_cast<float>(code) * scale);
}

__global__ void embed_gather_q8_kernel(const std::int32_t* ids, const std::uint8_t* codes,
                                       const std::uint8_t* scales, __nv_bfloat16* out,
                                       std::int32_t d, std::int32_t T, std::int32_t padded_d) {
    const std::int32_t kg     = padded_d / kEmbedGatherQ8Group;
    const std::int64_t n      = static_cast<std::int64_t>(d) * T;
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const std::int32_t t    = static_cast<std::int32_t>(i / d);
        const std::int32_t k    = static_cast<std::int32_t>(i - static_cast<std::int64_t>(t) * d);
        const std::int32_t row  = ids[t];
        const std::int32_t g    = k / kEmbedGatherQ8Group;
        const std::int32_t lane = k - g * kEmbedGatherQ8Group;
        const std::int64_t group_index = static_cast<std::int64_t>(row) * kg + g;
        const float scale              = common::load_fp16_unaligned(scales + group_index * 2);
        const auto code = static_cast<std::int8_t>(codes[group_index * kEmbedGatherQ8Group + lane]);
        out[i]          = __float2bfloat16(static_cast<float>(code) * scale);
    }
}

template <int Blocks, int Threads, bool PairStore>
__launch_bounds__(Threads) __global__
    void embed_gather_q8_packed_5120_kernel(const std::int32_t* ids, const std::uint8_t* codes,
                                            const std::uint8_t* scales, __nv_bfloat16* out) {
    constexpr int D = 5120, Values = D / Blocks, Words = Values / 4;
    static_assert(D % Blocks == 0 && Values % 32 == 0);
    const int token      = static_cast<int>(blockIdx.x) / Blocks;
    const int split      = static_cast<int>(blockIdx.x) % Blocks;
    const int row        = ids[token];
    const auto* code_row = codes + static_cast<std::int64_t>(row) * D;
    auto* output         = out + static_cast<std::int64_t>(token) * D;
#pragma unroll
    for (int item = threadIdx.x; item < Words; item += Threads) {
        const int d       = split * Values + item * 4;
        const auto group  = static_cast<std::int64_t>(row) * (D / 32) + d / 32;
        const float scale = common::load_fp16_unaligned(scales + group * 2);
        const auto word   = *reinterpret_cast<const std::uint32_t*>(code_row + d);
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
            const auto value =
                common::dequant_q8_pair(static_cast<std::uint16_t>(word >> (pair * 16)), scale);
            if constexpr (PairStore)
                reinterpret_cast<__nv_bfloat162*>(output + d)[pair] = value;
            else {
                output[d + pair * 2]     = value.x;
                output[d + pair * 2 + 1] = value.y;
            }
        }
    }
}

__launch_bounds__(32) __global__
    void embed_gather_q8_grouped_2048_kernel(const std::int32_t* ids, const std::uint8_t* codes,
                                             const std::uint8_t* scales, __nv_bfloat16* out) {
    const std::int32_t t   = static_cast<std::int32_t>(blockIdx.x) / kEmbedGatherQ8Groups;
    const std::int32_t g   = static_cast<std::int32_t>(blockIdx.x) - t * kEmbedGatherQ8Groups;
    const std::int32_t row = ids[t];
    const std::int64_t group_index = static_cast<std::int64_t>(row) * kEmbedGatherQ8Groups + g;

    int scale_bits = 0;
    if (threadIdx.x == 0) {
        scale_bits = static_cast<int>(scales[group_index * 2]) |
                     (static_cast<int>(scales[group_index * 2 + 1]) << 8);
    }
    scale_bits        = __shfl_sync(0xffffffffu, scale_bits, 0);
    const float scale = __half2float(__ushort_as_half(static_cast<std::uint16_t>(scale_bits)));
    const auto code   = static_cast<std::int8_t>(
        codes[group_index * kEmbedGatherQ8Group + static_cast<std::int32_t>(threadIdx.x)]);
    const std::int64_t out_idx = static_cast<std::int64_t>(t) * kEmbedGatherQ8D +
                                 static_cast<std::int64_t>(g) * kEmbedGatherQ8Group + threadIdx.x;
    out[out_idx] = __float2bfloat16(static_cast<float>(code) * scale);
}

__launch_bounds__(256) __global__
    void embed_gather_q8_row_2048_kernel(const std::int32_t* ids, const std::uint8_t* codes,
                                         const std::uint8_t* scales, __nv_bfloat16* out) {
    const int tid = static_cast<int>(threadIdx.x);
    const int t   = static_cast<int>(blockIdx.x);
    const int row = ids[t];

    constexpr int kValuesPerThread = kEmbedGatherQ8D / 256;
    static_assert(kValuesPerThread == 8);
    const int k = tid * kValuesPerThread;
    const std::int64_t group_index =
        static_cast<std::int64_t>(row) * kEmbedGatherQ8Groups + k / kEmbedGatherQ8Group;
    const float scale    = common::load_fp16_unaligned(scales + group_index * 2);
    const auto* code_row = codes + static_cast<std::int64_t>(row) * kEmbedGatherQ8D;
    const uint2 packed   = *reinterpret_cast<const uint2*>(code_row + k);
    auto* out_pairs =
        reinterpret_cast<__nv_bfloat162*>(out + static_cast<std::int64_t>(t) * kEmbedGatherQ8D + k);
#pragma unroll
    for (int word_index = 0; word_index < 2; ++word_index) {
        const std::uint32_t word = word_index == 0 ? packed.x : packed.y;
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
            out_pairs[word_index * 2 + pair] =
                common::dequant_q8_pair(static_cast<std::uint16_t>(word >> (pair * 16)), scale);
        }
    }
}

} // namespace ninfer::ops::detail::embedding
