#pragma once

#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/quant/integer.cuh"
#include "ops/kv_cache/kernels/layout.cuh"
#include "ops/kv_cache/kernels/hadamard.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace ninfer::ops::detail::kv {
template <typename Geometry, typename Metadata, bool MultiBatch = false>
__launch_bounds__(256) __global__
    void write_i8_kernel(const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
                         const std::int32_t* __restrict__ positions, Metadata metadata,
                         std::int8_t* __restrict__ cache_k, std::int8_t* __restrict__ cache_v,
                         __half* __restrict__ scale_k, __half* __restrict__ scale_v,
                         std::int32_t width) {
    if constexpr (MultiBatch) {
        const int batch = blockIdx.z;
        metadata.table_rows += batch;
        if (metadata.valid_columns) metadata.valid_columns += batch;
        const auto offset = static_cast<std::int64_t>(batch) * width * 256 * Geometry::KVHeads;
        k += offset;
        v += offset;
        positions += batch * width;
    }
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffu;
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
    std::int64_t page               = lane == 0 ? lookup_page(block_table, position) : 0;
    const int page_off              = position & kPageMask;
    page                            = __shfl_sync(FullMask, page, 0);

    float k_values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        k_values[r] = __bfloat162float(k[source_offset<Geometry>(kv_head, d, token)]);
    }
    normalized_hadamard_d256_inplace(k_values, lane);

#pragma unroll
    for (int group = 0; group < kInt8Groups; ++group) {
        const int d0            = group * kInt8Group + lane;
        const int d1            = d0 + 32;
        const float k0          = k_values[2 * group];
        const float k1          = k_values[2 * group + 1];
        const std::int64_t src0 = source_offset<Geometry>(kv_head, d0, token);
        const std::int64_t src1 = source_offset<Geometry>(kv_head, d1, token);
        const float v0          = __bfloat162float(v[src0]);
        const float v1          = __bfloat162float(v[src1]);
        const float k_abs       = common::warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMask);
        const float v_abs       = common::warp_max(fmaxf(fabsf(v0), fabsf(v1)), FullMask);
        const auto k_quant      = common::int8_fp16_scale(k_abs);
        const auto v_quant      = common::int8_fp16_scale(v_abs);
        const std::int64_t code_base =
            Int8G64Layout::code_offset(page, kv_head, group * kInt8Group, page_off);
        cache_k[code_base + lane]      = common::int8_quant_code(k0, k_quant.inverse_scale);
        cache_k[code_base + lane + 32] = common::int8_quant_code(k1, k_quant.inverse_scale);
        cache_v[code_base + lane]      = common::int8_quant_code(v0, v_quant.inverse_scale);
        cache_v[code_base + lane + 32] = common::int8_quant_code(v1, v_quant.inverse_scale);
        if (lane == 0) {
            const std::int64_t scale_off =
                Int8G64Layout::scale_offset(page, kv_head, group, page_off);
            scale_k[scale_off] = k_quant.scale;
            scale_v[scale_off] = v_quant.scale;
        }
    }
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void write_i8_page_kernel(const __nv_bfloat16* __restrict__ k,
                              const __nv_bfloat16* __restrict__ v,
                              const std::int32_t* __restrict__ positions, Metadata metadata,
                              std::int8_t* __restrict__ cache_k, std::int8_t* __restrict__ cache_v,
                              __half* __restrict__ scale_k, __half* __restrict__ scale_v,
                              std::int32_t width) {
    constexpr int TokensPerTile = 8;
    constexpr unsigned FullMask = 0xffffffffu;
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

    const int position = base_position + token;
    const int page_off = position & kPageMask;

    float k_values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        k_values[r] = __bfloat162float(k[source_offset<Geometry>(kv_head, d, token)]);
    }
    normalized_hadamard_d256_inplace(k_values, lane);

#pragma unroll
    for (int group = 0; group < kInt8Groups; ++group) {
        const int d0            = group * kInt8Group + lane;
        const int d1            = d0 + 32;
        const float k0          = k_values[2 * group];
        const float k1          = k_values[2 * group + 1];
        const std::int64_t src0 = source_offset<Geometry>(kv_head, d0, token);
        const std::int64_t src1 = source_offset<Geometry>(kv_head, d1, token);
        const float v0          = __bfloat162float(v[src0]);
        const float v1          = __bfloat162float(v[src1]);
        const float k_abs       = common::warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMask);
        const float v_abs       = common::warp_max(fmaxf(fabsf(v0), fabsf(v1)), FullMask);
        const auto k_quant      = common::int8_fp16_scale(k_abs);
        const auto v_quant      = common::int8_fp16_scale(v_abs);
        const std::int64_t code_base =
            Int8G64Layout::code_offset(record, kv_head, group * kInt8Group, page_off);
        cache_k[code_base + lane]      = common::int8_quant_code(k0, k_quant.inverse_scale);
        cache_k[code_base + lane + 32] = common::int8_quant_code(k1, k_quant.inverse_scale);
        cache_v[code_base + lane]      = common::int8_quant_code(v0, v_quant.inverse_scale);
        cache_v[code_base + lane + 32] = common::int8_quant_code(v1, v_quant.inverse_scale);
        if (lane == 0) {
            const std::int64_t scale_offset =
                Int8G64Layout::scale_offset(record, kv_head, group, page_off);
            scale_k[scale_offset] = k_quant.scale;
            scale_v[scale_offset] = v_quant.scale;
        }
    }
}

} // namespace ninfer::ops::detail::kv
