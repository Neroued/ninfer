#pragma once

#include "ops/softmax_attention/kernels/context_query/kernel.cuh"

namespace ninfer::ops::detail {

struct SlidingWindowAttentionPolicy {
    static constexpr bool PageMapped = false;

    const std::int32_t* positions;
    int padded_context;
    int window_mask;

    __device__ __forceinline__ int context_count(int value) const {
        return min(value, window_mask);
    }

    __device__ __forceinline__ int query_position(int token) const { return positions[token]; }

    __device__ __forceinline__ bool allow_context(int query, int key) const {
        return key >= query - (window_mask);
    }

    __device__ __forceinline__ std::int64_t context_tile(int kv_head, int, std::int64_t) const {
        return static_cast<std::int64_t>(kContextQueryHeadDim) * padded_context * kv_head;
    }

    __device__ __forceinline__ std::int64_t context_index(std::int64_t tile, int d, int position,
                                                          int) const {
        const int slot = position & (window_mask);
        return tile + d + static_cast<std::int64_t>(kContextQueryHeadDim) * slot;
    }

    template <int Tokens, int KeyBlock>
    __device__ __forceinline__ void prime(int, int, int, int) {}
};

template <int Tokens, int WarpsPerCta, int KeyBlock, bool DirectOutput, int KVHeads>
__launch_bounds__(WarpsPerCta * 32, 2) __global__
    void sliding_window_attention_split_partial_kernel(
        const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ query_k,
        const __nv_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ positions,
        const std::int32_t* __restrict__ valid_columns,
        const std::int64_t* __restrict__ key_offsets,
        const std::int64_t* __restrict__ value_offsets, const __nv_bfloat16* __restrict__ context_k,
        const __half* __restrict__ context_v, int padded_context, int window_mask, int max_context,
        int split_capacity, float scale, float* __restrict__ partial_acc,
        float* __restrict__ partial_m, float* __restrict__ partial_l,
        __nv_bfloat16* __restrict__ out) {
    const int batch = static_cast<int>(blockIdx.z);
    if (valid_columns[batch] > 0) {
        context_k += key_offsets[batch] / 2;
        context_v += value_offsets[batch] / 2;
    }
    const std::int32_t* batch_positions = positions + static_cast<std::int64_t>(Tokens) * batch;
    SlidingWindowAttentionPolicy policy{
        .positions      = batch_positions,
        .padded_context = padded_context,
        .window_mask    = window_mask,
    };
    context_query_split_partial_body<SlidingWindowAttentionPolicy, Tokens, WarpsPerCta, KeyBlock,
                                     DirectOutput, float, KVHeads>(
        q, query_k, query_v, valid_columns, context_k, context_v, policy,
        valid_columns[batch] == 0 ? 0 : batch_positions[0], max_context, split_capacity, scale,
        partial_acc, partial_m, partial_l, out);
}

template <int Tokens, int KeyBlock, int WarpsPerBlock, int KVHeads>
__launch_bounds__(WarpsPerBlock * 32, 2) __global__ void sliding_window_attention_reduce_kernel(
    const float* __restrict__ partial_acc, const float* __restrict__ partial_m,
    const float* __restrict__ partial_l, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, int window_mask, int max_context,
    int split_capacity, __nv_bfloat16* __restrict__ out) {
    const int batch  = blockIdx.z;
    const int length = valid_columns[batch] > 0 ? positions[std::int64_t(Tokens) * batch] : 0;
    context_query_reduce_body<Tokens, KeyBlock, WarpsPerBlock, KVHeads, 32>(
        partial_acc, partial_m, partial_l, length, min(length, window_mask), valid_columns,
        max_context, split_capacity, out);
}

} // namespace ninfer::ops::detail
