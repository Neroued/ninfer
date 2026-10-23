#pragma once

#include "ops/softmax_attention/kernels/context_query/kernel.cuh"
#include "ops/kv_cache/kernels/layout.cuh"

namespace ninfer::ops::detail {

struct ContextAttentionPolicy {
    static constexpr bool PageMapped = true;

    const std::int64_t* block_table;
    int logical_pages;
    int table_group                = -1;
    std::int64_t table_lane_offset = 0;

    __device__ __forceinline__ int context_count(int value) const { return value; }

    __device__ __forceinline__ int query_position(int) const { return 0; }

    __device__ __forceinline__ bool allow_context(int, int) const { return true; }

    __device__ __forceinline__ std::int64_t context_tile(int kv_head, int key0,
                                                         std::int64_t page_offset) const {
        return kv::record_offset<kContextQueryHeadDim, 2>(page_offset, kv_head,
                                                          key0 & kv::kPageMask, 0);
    }

    __device__ __forceinline__ std::int64_t context_index(std::int64_t tile, int d, int,
                                                          int row) const {
        return tile + d + static_cast<std::int64_t>(kContextQueryHeadDim) * row;
    }

    template <int Tokens, int KeyBlock>
    __device__ __forceinline__ void prime(int context_tile_count, int context_start, int tile_begin,
                                          int lane) {
        if constexpr (Tokens == 4) {
            if (context_tile_count > 0) {
                const int first_logical_page = (context_start + tile_begin * KeyBlock) >> 6;
                const int first_group        = first_logical_page & ~3;
                const int table_index        = first_group + lane;
                if (lane < 4 && table_index < logical_pages) {
                    table_lane_offset = __ldg(block_table + table_index);
                }
                table_group = first_group;
            }
        }
    }

    template <int Tokens>
    __device__ __forceinline__ std::int64_t page(int key0, int lane, unsigned mask) {
        const int logical_page = key0 >> 6;
        if constexpr (Tokens == 4) {
            const int group = logical_page & ~3;
            if (group != table_group) {
                const int table_index = group + lane;
                if (lane < 4 && table_index < logical_pages) {
                    table_lane_offset = __ldg(block_table + table_index);
                }
                table_group = group;
            }
            return __shfl_sync(mask, table_lane_offset, logical_page - group);
        } else {
            const std::int64_t record_offset = lane == 0 ? __ldg(block_table + logical_page) : 0;
            return __shfl_sync(mask, record_offset, 0);
        }
    }
};

template <int Tokens, int WarpsPerCta, int KeyBlock, bool DirectOutput>
__launch_bounds__(WarpsPerCta * 32, 2) __global__ void context_attention_split_partial_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ query_k,
    const __nv_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ context_length,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ table_rows,
    const __nv_bfloat16* __restrict__ context_k, const __half* __restrict__ context_v,
    const std::int64_t* __restrict__ block_tables, int logical_pages, int max_context,
    int split_capacity, float scale, __nv_bfloat16* __restrict__ partial_acc,
    float* __restrict__ partial_m, float* __restrict__ partial_l, __nv_bfloat16* __restrict__ out) {
    const int batch = static_cast<int>(blockIdx.z);
    ContextAttentionPolicy policy{
        .block_table = block_tables + static_cast<std::int64_t>(logical_pages) * table_rows[batch],
        .logical_pages = logical_pages,
    };
    context_query_split_partial_body<ContextAttentionPolicy, Tokens, WarpsPerCta, KeyBlock,
                                     DirectOutput>(
        q, query_k, query_v, valid_columns, context_k, context_v, policy, context_length[batch],
        max_context, split_capacity, scale, partial_acc, partial_m, partial_l, out);
}

template <int Tokens, int KeyBlock>
__launch_bounds__(128, 2) __global__
    void context_attention_reduce_kernel(const __nv_bfloat16* __restrict__ partial_acc,
                                         const float* __restrict__ partial_m,
                                         const float* __restrict__ partial_l,
                                         const std::int32_t* __restrict__ context_length,
                                         const std::int32_t* __restrict__ valid_columns,
                                         int max_context, int split_capacity,
                                         __nv_bfloat16* __restrict__ out) {
    const int length = context_length[static_cast<int>(blockIdx.z)];
    context_query_reduce_body<Tokens, KeyBlock>(partial_acc, partial_m, partial_l, length, length,
                                                valid_columns, max_context, split_capacity, out);
}

} // namespace ninfer::ops::detail
