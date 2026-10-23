#pragma once

#include "ops/common/cuda/memory.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/kv_cache/kernels/layout.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace ninfer::ops::detail {
inline constexpr int kKVCacheAppendPrefixHeadDim = 128;
inline constexpr int kKVCacheAppendPrefixHeads   = 8;
inline constexpr int kKVCacheAppendPrefixPage    = 64;

template <int Capacity>
__device__ __forceinline__ void
prefix_copy_cyclic_unit(const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
                        __nv_bfloat16* __restrict__ cache_k, __half* __restrict__ cache_v,
                        int token, int unit_in_token, int slot, int padded_capacity) {
    constexpr int ElementsPerUnit = 8;
    constexpr int UnitsPerHead    = kKVCacheAppendPrefixHeadDim / ElementsPerUnit;
    const int kv_head             = unit_in_token / UnitsPerHead;
    const int d                   = (unit_in_token - kv_head * UnitsPerHead) * ElementsPerUnit;
    const std::int64_t src =
        static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kKVCacheAppendPrefixHeadDim) *
                                           (kv_head + kKVCacheAppendPrefixHeads * token);
    const std::int64_t dst = kv::cyclic_offset<128, Capacity>(slot, kv_head, padded_capacity, d);

    const int4 key   = *reinterpret_cast<const int4*>(&k[src]);
    const int4 value = common::bf16x8_bits_to_f16x8_bits(*reinterpret_cast<const int4*>(&v[src]));
    *reinterpret_cast<int4*>(&cache_k[dst]) = key;
    *reinterpret_cast<int4*>(&cache_v[dst]) = value;
}

__device__ __forceinline__ void
prefix_copy_paged_unit(const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
                       __nv_bfloat16* __restrict__ cache_k, __half* __restrict__ cache_v, int token,
                       int unit_in_token, int page_offset, std::int64_t record_offset) {
    constexpr int ElementsPerUnit = 16;
    constexpr int UnitsPerHead    = kKVCacheAppendPrefixHeadDim / ElementsPerUnit;
    const int kv_head             = unit_in_token / UnitsPerHead;
    const int d                   = (unit_in_token - kv_head * UnitsPerHead) * ElementsPerUnit;
    const std::int64_t src =
        static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kKVCacheAppendPrefixHeadDim) *
                                           (kv_head + kKVCacheAppendPrefixHeads * token);
    const std::int64_t dst =
        kv::record_offset<kKVCacheAppendPrefixHeadDim, 2>(record_offset, kv_head, page_offset, d);

    const int4 k0 = *reinterpret_cast<const int4*>(&k[src]);
    const int4 v0 = common::bf16x8_bits_to_f16x8_bits(*reinterpret_cast<const int4*>(&v[src]));
    *reinterpret_cast<int4*>(&cache_k[dst]) = k0;
    *reinterpret_cast<int4*>(&cache_v[dst]) = v0;
    const int4 k1                           = *reinterpret_cast<const int4*>(&k[src + 8]);
    const int4 v1 = common::bf16x8_bits_to_f16x8_bits(*reinterpret_cast<const int4*>(&v[src + 8]));
    *reinterpret_cast<int4*>(&cache_k[dst + 8]) = k1;
    *reinterpret_cast<int4*>(&cache_v[dst + 8]) = v1;
}

template <int Capacity, int Threads>
__global__ void prefix_cyclic_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ counts,
    const std::int64_t* __restrict__ state_offsets, __nv_bfloat16* __restrict__ cache_k,
    __half* __restrict__ cache_v, int min_count, int max_count, int width, int padded_capacity) {
    static_assert(Capacity == 2048 || Capacity == 4096);
    static_assert((Capacity & (Capacity - 1)) == 0);
    constexpr int UnitsPerToken = kKVCacheAppendPrefixHeads * kKVCacheAppendPrefixHeadDim / 8;
    const int unit              = static_cast<int>(blockIdx.x) * Threads + threadIdx.x;
    const int token             = unit / UnitsPerToken;
    const int unit_in_token     = unit % UnitsPerToken;
    const int batch             = static_cast<int>(blockIdx.y);
    const int count             = counts[batch];
    if (count < min_count || count > max_count || token >= count) return;

    constexpr std::int64_t ElementsPerToken =
        kKVCacheAppendPrefixHeadDim * kKVCacheAppendPrefixHeads;
    const std::int64_t input_offset = ElementsPerToken * width * batch;
    k += input_offset;
    v += input_offset;
    positions += static_cast<std::int64_t>(width) * batch;
    cache_k += state_offsets[batch] / sizeof(__nv_bfloat16);
    cache_v += state_offsets[batch] / sizeof(__half);

    const int position = positions[token];
    const int slot     = position & (Capacity - 1);
    prefix_copy_cyclic_unit<Capacity>(k, v, cache_k, cache_v, token, unit_in_token, slot,
                                      padded_capacity);
}

__global__ void prefix_paged_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ counts,
    const std::int32_t* __restrict__ table_rows, __nv_bfloat16* __restrict__ cache_k,
    __half* __restrict__ cache_v, const std::int64_t* __restrict__ block_tables, int logical_pages,
    int min_count, int max_count, int width) {
    constexpr int UnitsPerToken  = kKVCacheAppendPrefixHeads * 8;
    constexpr int TokensPerBlock = 256 / UnitsPerToken;
    static_assert(TokensPerBlock * UnitsPerToken == 256);
    const int batch = static_cast<int>(blockIdx.y);
    const int count = counts[batch];
    if (count == 0 || count < min_count || count > max_count) return;

    constexpr std::int64_t ElementsPerToken =
        kKVCacheAppendPrefixHeadDim * kKVCacheAppendPrefixHeads;
    const std::int64_t input_offset = ElementsPerToken * width * batch;
    k += input_offset;
    v += input_offset;
    positions += static_cast<std::int64_t>(width) * batch;
    const std::int64_t* block_table =
        block_tables + static_cast<std::int64_t>(logical_pages) * table_rows[batch];

    const int local            = static_cast<int>(threadIdx.x);
    const int local_token      = local / UnitsPerToken;
    const int unit_in_token    = local - local_token * UnitsPerToken;
    const int lane             = local & 31;
    const int token            = static_cast<int>(blockIdx.x) * TokensPerBlock + local_token;
    int position               = 0;
    std::int64_t record_offset = 0;
    if (lane == 0 && token < count) {
        position      = positions[token];
        record_offset = block_table[position >> 6];
    }
    position      = __shfl_sync(0xffffffffu, position, 0);
    record_offset = __shfl_sync(0xffffffffu, record_offset, 0);
    if (token < count) {
        prefix_copy_paged_unit(k, v, cache_k, cache_v, token, unit_in_token,
                               position & (kKVCacheAppendPrefixPage - 1), record_offset);
    }
}

} // namespace ninfer::ops::detail
