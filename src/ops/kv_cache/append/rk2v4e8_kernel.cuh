#pragma once

// rk2v4-e8 append kernel. K receives the fixed normalized D256 rotation and the E8-root
// cylinder codec (quarter width); V receives the per-64-group Hadamard rotation and the packed
// int4 codec (half width). Both planes carry a per-64-group FP16 scale. The row body is shared
// with the fused append inside the attention kernels so both paths emit identical codes.

#include "ops/kv_cache/append/geometry.cuh"
#include "ops/kv_cache/rk2v4e8_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__ void kv_cache_append_full_rk2v4e8_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, Metadata metadata,
    std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    __half* __restrict__ scale_k, __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int unit              = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units             = tokens * Geometry::KVHeads;
    if (unit >= units) return;

    const int kv_head               = unit % Geometry::KVHeads;
    const int token                 = unit / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    physical_page     = __shfl_sync(FullMask, physical_page, 0);
    rk2v4e8_append_row<Geometry>(k, v, cache_k, cache_v, scale_k, scale_v, token, kv_head,
                                 physical_page, position & kPagedKVPageMask, lane);
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__ void kv_cache_append_full_rk2v4e8_page_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, Metadata metadata,
    std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    __half* __restrict__ scale_k, __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int TokensPerTile = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int kv_head           = static_cast<int>(blockIdx.y);
    const int base_position     = positions[0];
    const int tile_position =
        (base_position / TokensPerTile + static_cast<int>(blockIdx.x)) * TokensPerTile;
    const int token_begin = max(0, tile_position - base_position);
    const int token_end   = min(tokens, tile_position + TokensPerTile - base_position);
    const int token       = token_begin + warp;
    if (token >= token_end) return;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page  = lane == 0 ? block_table[tile_position >> kPagedKVPageShift] : 0;
    physical_page      = __shfl_sync(FullMask, physical_page, 0);
    const int position = base_position + token;
    rk2v4e8_append_row<Geometry>(k, v, cache_k, cache_v, scale_k, scale_v, token, kv_head,
                                 physical_page, position & kPagedKVPageMask, lane);
}

} // namespace ninfer::ops
