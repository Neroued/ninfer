#pragma once

#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/quant/fp8.cuh"
#include "ops/kv_cache/kernels/layout.cuh"
#include "ops/kv_cache/kernels/hadamard.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace ninfer::ops::detail::kv {
template <typename Geometry>
__device__ __forceinline__ void
write_fp8_row(const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
              std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
              __half* __restrict__ scale_k, __half* __restrict__ scale_v, int token, int kv_head,
              std::int64_t record, int page_off, int lane) {
    constexpr unsigned FullMask = 0xffffffffU;
    float values[8];
    float local_absmax = 0.0F;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        values[r]   = __bfloat162float(k[source_offset<Geometry>(kv_head, d, token)]);
    }
    normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (float value : values) { local_absmax = fmaxf(local_absmax, fabsf(value)); }
    const auto k_quant = common::fp8_fp16_scale(common::warp_max(local_absmax, FullMask));
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        cache_k[Fp8RowLayout::code_offset(record, kv_head, d, page_off)] =
            common::fp8_quant_code(values[r], k_quant.inverse_scale);
    }
    if (lane == 0) {
        scale_k[Fp8RowLayout::scale_offset(record, kv_head, page_off)] = k_quant.scale;
    }

    local_absmax = 0.0F;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d  = lane + 32 * r;
        values[r]    = __bfloat162float(v[source_offset<Geometry>(kv_head, d, token)]);
        local_absmax = fmaxf(local_absmax, fabsf(values[r]));
    }
    const auto v_quant = common::fp8_fp16_scale(common::warp_max(local_absmax, FullMask));
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        cache_v[Fp8RowLayout::code_offset(record, kv_head, d, page_off)] =
            common::fp8_quant_code(values[r], v_quant.inverse_scale);
    }
    if (lane == 0) {
        scale_v[Fp8RowLayout::scale_offset(record, kv_head, page_off)] = v_quant.scale;
    }
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void write_fp8_kernel(const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
                          const std::int32_t* __restrict__ positions, Metadata metadata,
                          std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
                          __half* __restrict__ scale_k, __half* __restrict__ scale_v,
                          std::int32_t width) {
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
    const std::int64_t* block_table = metadata.block_table();
    std::int64_t record             = lane == 0 ? lookup_page(block_table, position) : 0;
    record                          = __shfl_sync(FullMask, record, 0);
    write_fp8_row<Geometry>(k, v, cache_k, cache_v, scale_k, scale_v, token, kv_head, record,
                            position & kPageMask, lane);
}

// Independent request rows share physical planes, but select separate input and table rows.
template <typename Geometry, bool Masked>
__launch_bounds__(256) __global__
    void write_fp8_batch_kernel(const __nv_bfloat16* k, const __nv_bfloat16* v,
                                const std::int32_t* positions, BatchMetadata<Masked> metadata,
                                std::uint8_t* cache_k, std::uint8_t* cache_v, __half* scale_k,
                                __half* scale_v, std::int32_t width) {
    const int batch = blockIdx.z;
    metadata.table_rows += batch;
    if constexpr (Masked) metadata.valid_columns += batch;
    const int tokens = metadata.valid_tokens(width);
    const int unit   = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (unit >= tokens * Geometry::KVHeads) return;
    const int token         = unit / Geometry::KVHeads;
    const int kv_head       = unit % Geometry::KVHeads;
    const int lane          = threadIdx.x & 31;
    const auto input_offset = static_cast<std::int64_t>(batch) * width * 256 * Geometry::KVHeads;
    const int position      = positions[batch * width] + token;
    const auto* table       = metadata.block_table();
    std::int64_t page       = lane == 0 ? lookup_page(table, position) : 0;
    page                    = __shfl_sync(0xffffffffU, page, 0);
    write_fp8_row<Geometry>(k + input_offset, v + input_offset, cache_k, cache_v, scale_k, scale_v,
                            token, kv_head, page, position & kPageMask, lane);
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void write_fp8_page_kernel(const __nv_bfloat16* __restrict__ k,
                               const __nv_bfloat16* __restrict__ v,
                               const std::int32_t* __restrict__ positions, Metadata metadata,
                               std::uint8_t* __restrict__ cache_k,
                               std::uint8_t* __restrict__ cache_v, __half* __restrict__ scale_k,
                               __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int TokensPerTile = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int kv_head           = static_cast<int>(blockIdx.y);
    const int tile_delta        = static_cast<int>(blockIdx.x);
    const int base_position     = positions[0];
    const int tile_position     = (base_position / TokensPerTile + tile_delta) * TokensPerTile;
    const int logical_page      = tile_position >> kPageShift;
    const int token_begin       = max(0, tile_position - base_position);
    const int token_end         = min(tokens, tile_position + TokensPerTile - base_position);
    if (token_begin >= token_end) return;

    const int token = token_begin + warp;
    if (token >= token_end) return;
    const std::int64_t* block_table = metadata.block_table();
    std::int64_t record             = lane == 0 ? block_table[logical_page] : 0;
    record                          = __shfl_sync(FullMask, record, 0);
    const int position              = base_position + token;
    write_fp8_row<Geometry>(k, v, cache_k, cache_v, scale_k, scale_v, token, kv_head, record,
                            position & kPageMask, lane);
}

} // namespace ninfer::ops::detail::kv
