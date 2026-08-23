#pragma once

#include "ops/kernel/bidirectional_gqa_attention.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

// SM70 fallback for DFlash full-context and sliding-window attention. One CTA
// owns one (query head, proposal token, KV split). The 128 lanes reduce Q.K and
// independently accumulate one output channel with an online softmax, retaining
// the public split/reduce workspace contract used by the tensor-core launcher.
template <bool CyclicSwa, int Tokens, int KeyBlock, bool DirectOutput>
__launch_bounds__(128, 2) __global__ void noncausal_gqa_volta_partial_kernel(
        const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ query_k,
        const __nv_bfloat16* __restrict__ query_v,
        const std::int32_t* __restrict__ context_state,
        const std::int32_t* __restrict__ valid_columns,
        const std::int32_t* __restrict__ selectors,
        const __nv_bfloat16* __restrict__ context_k,
        const __nv_bfloat16* __restrict__ context_v,
        const std::int32_t* __restrict__ block_tables, int context_stride, int logical_pages,
        int max_context, int split_capacity, float scale,
        __nv_bfloat16* __restrict__ partial_acc, float* __restrict__ partial_m,
        float* __restrict__ partial_l, __nv_bfloat16* __restrict__ out) {
    static_assert(Tokens >= 1 && Tokens <= 16);
    static_assert(KeyBlock == 32 || KeyBlock == 64);

    constexpr int D = kBidirectionalGqaHeadDim;
    const int d     = static_cast<int>(threadIdx.x);
    const int lane  = d & 31;
    const int warp  = d >> 5;
    const int row   = static_cast<int>(blockIdx.x);
    const int token = row / kBidirectionalGqaQHeads;
    const int q_head = row - token * kBidirectionalGqaQHeads;
    const int kv_head = q_head / kBidirectionalGqaGroup;
    const int split   = static_cast<int>(blockIdx.y);
    const int batch   = static_cast<int>(blockIdx.z);

    constexpr std::int64_t QueryElements =
        static_cast<std::int64_t>(D) * kBidirectionalGqaQHeads * Tokens;
    constexpr std::int64_t QueryKvElements =
        static_cast<std::int64_t>(D) * kBidirectionalGqaKVHeads * Tokens;
    constexpr std::int64_t StatElements =
        static_cast<std::int64_t>(kBidirectionalGqaQHeads) * Tokens;
    q += QueryElements * batch;
    query_k += QueryKvElements * batch;
    query_v += QueryKvElements * batch;
    out += QueryElements * batch;
    partial_acc += QueryElements * split_capacity * batch;
    partial_m += StatElements * split_capacity * batch;
    partial_l += StatElements * split_capacity * batch;

    int length;
    if constexpr (CyclicSwa) {
        context_state += static_cast<std::int64_t>(Tokens) * batch;
        length = context_state[0];
        const std::int64_t lane_elements =
            static_cast<std::int64_t>(D) * context_stride * kBidirectionalGqaKVHeads;
        context_k += lane_elements * selectors[batch];
        context_v += lane_elements * selectors[batch];
    } else {
        context_state += batch;
        length = context_state[0];
        block_tables += static_cast<std::int64_t>(logical_pages) * selectors[batch];
    }

    const int valid = valid_columns[batch];
    if (token >= Tokens || split >= split_capacity || length < 0 || length > max_context ||
        valid < 1 || valid > Tokens) {
        return;
    }
    if (token >= valid) {
        if constexpr (DirectOutput) {
            out[bidirectional_gqa_q_index(q_head, d, token)] = __float2bfloat16(0.0f);
        }
        return;
    }

    const int context_count = CyclicSwa ? min(length, kSwaWindow - 1) : length;
    const int context_start = length - context_count;
    const int context_tiles = (context_count + KeyBlock - 1) / KeyBlock;
    const int active_splits = context_tiles > 0 ? min(context_tiles, split_capacity) : 1;
    if (split >= active_splits) { return; }
    const int tile_begin =
        static_cast<int>((static_cast<std::int64_t>(context_tiles) * split) / active_splits);
    const int tile_end = static_cast<int>(
        (static_cast<std::int64_t>(context_tiles) * (split + 1)) / active_splits);
    const int key_begin = context_start + tile_begin * KeyBlock;
    const int key_end   = min(length, context_start + tile_end * KeyBlock);
    const bool owns_query = split == active_splits - 1;
    const int q_position  = CyclicSwa ? context_state[token] : 0;

    const float q_value =
        __bfloat162float(q[bidirectional_gqa_q_index(q_head, d, token)]);
    float numerator = 0.0f;
    float m         = -CUDART_INF_F;
    float l         = 0.0f;
    __shared__ float warp_sums[4];
    __shared__ float step_alpha;
    __shared__ float step_probability;
    __shared__ float final_m;
    __shared__ float final_l;
    if (d == 0) {
        final_m = -CUDART_INF_F;
        final_l = 0.0f;
    }
    __syncthreads();

    const auto consume = [&](float k_value, float v_value) {
        float dot = warp_sum<32>(q_value * k_value, 0xffffffffu);
        if (lane == 0) { warp_sums[warp] = dot; }
        __syncthreads();
        if (d == 0) {
            const float score = (warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3]) * scale;
            const float next_m = fmaxf(m, score);
            step_alpha = m == -CUDART_INF_F ? 0.0f : expf(m - next_m);
            step_probability = expf(score - next_m);
            m = next_m;
            l = l * step_alpha + step_probability;
            final_m = m;
            final_l = l;
        }
        __syncthreads();
        numerator = numerator * step_alpha + step_probability * v_value;
        __syncthreads();
    };

    for (int key = key_begin; key < key_end; ++key) {
        if constexpr (CyclicSwa) {
            if (key < q_position - (kSwaWindow - 1)) { continue; }
            const auto index = bidirectional_gqa_cyclic_context_index(
                kv_head, d, key & (kSwaWindow - 1), context_stride);
            consume(__bfloat162float(context_k[index]), __bfloat162float(context_v[index]));
        } else {
            const int logical_page  = key >> 6;
            const int physical_page = block_tables[logical_page];
            const std::int64_t index = static_cast<std::int64_t>(d) +
                                       static_cast<std::int64_t>(D) *
                                           ((key & 63) + 64 * (physical_page + context_stride * kv_head));
            consume(__bfloat162float(context_k[index]), __bfloat162float(context_v[index]));
        }
    }
    if (owns_query) {
        for (int key = 0; key < valid; ++key) {
            const auto index = bidirectional_gqa_query_kv_index(kv_head, d, key);
            consume(__bfloat162float(query_k[index]), __bfloat162float(query_v[index]));
        }
    }

    if constexpr (DirectOutput) {
        const float value = final_l > 0.0f ? numerator / final_l : 0.0f;
        out[bidirectional_gqa_q_index(q_head, d, token)] = __float2bfloat16(value);
    } else {
        partial_acc[bidirectional_gqa_partial_index<Tokens>(q_head, d, token, split)] =
            __float2bfloat16(numerator);
        if (d == 0) {
            const auto stat = bidirectional_gqa_stat_index<Tokens>(q_head, token, split);
            partial_m[stat] = final_m;
            partial_l[stat] = final_l;
        }
    }
}

} // namespace ninfer::ops
