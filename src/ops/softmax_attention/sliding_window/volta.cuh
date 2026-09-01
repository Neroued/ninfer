#pragma once

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/softmax_attention/common/context_query.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

namespace ninfer::ops {

inline constexpr int kSlidingWindowVoltaThreads = kContextQueryHeadDim;

__launch_bounds__(kSlidingWindowVoltaThreads, 1) __global__ void
sliding_window_attention_volta_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ query_k,
    const __nv_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ lanes,
    const __nv_bfloat16* __restrict__ context_k,
    const __nv_bfloat16* __restrict__ context_v, int padded_context, int tokens, float scale,
    __nv_bfloat16* __restrict__ out) {
    constexpr int D       = kContextQueryHeadDim;
    constexpr int QHeads  = kContextQueryQHeads;
    constexpr int KVHeads = kContextQueryKVHeads;
    constexpr int Window  = 4096;

    __shared__ float warp_sums[kSlidingWindowVoltaThreads / kWarpSize];
    __shared__ float score_s;

    const int token  = static_cast<int>(blockIdx.x);
    const int q_head = static_cast<int>(blockIdx.y);
    const int batch  = static_cast<int>(blockIdx.z);
    const int d      = static_cast<int>(threadIdx.x);
    const int valid  = valid_columns[batch];
    const std::int64_t q_batch = static_cast<std::int64_t>(batch) * D * QHeads * tokens;
    const std::int64_t kv_batch = static_cast<std::int64_t>(batch) * D * KVHeads * tokens;
    const std::int64_t out_index =
        q_batch + d + static_cast<std::int64_t>(D) * (q_head + QHeads * token);
    if (token >= valid) {
        out[out_index] = __float2bfloat16(0.0f);
        return;
    }

    const auto* batch_positions = positions + static_cast<std::int64_t>(batch) * tokens;
    const int context_end        = batch_positions[0];
    const int query_position     = batch_positions[token];
    const int context_begin      = max(0, max(context_end - Window, query_position - (Window - 1)));
    const int kv_head            = q_head / (QHeads / KVHeads);
    const float q_value          = __bfloat162float(q[out_index]);

    const std::int64_t lane_base =
        static_cast<std::int64_t>(lanes[batch]) * D * padded_context * KVHeads;
    float row_max = -CUDART_INF_F;
    float row_sum = 0.0f;
    float acc     = 0.0f;

    for (int key = context_begin; key < context_end; ++key) {
        const int slot = key & (Window - 1);
        const std::int64_t index = lane_base + d + static_cast<std::int64_t>(D) *
                                                       (slot + padded_context * kv_head);
        const float key_value = __bfloat162float(context_k[index]);
        const float value     = __bfloat162float(context_v[index]);
        const float dot = block_reduce_sum<kSlidingWindowVoltaThreads>(q_value * key_value,
                                                                       warp_sums);
        if (d == 0) { score_s = dot * scale; }
        __syncthreads();
        const float score      = score_s;
        const float next_max   = fmaxf(row_max, score);
        const float old_scale  = row_max == -CUDART_INF_F
                                     ? 0.0f
                                     : exp2_approx((row_max - next_max) * 1.4426950408889634f);
        const float probability = exp2_approx((score - next_max) * 1.4426950408889634f);
        acc     = acc * old_scale + probability * value;
        row_sum = row_sum * old_scale + probability;
        row_max = next_max;
        __syncthreads();
    }

    for (int key_token = 0; key_token < valid; ++key_token) {
        const int key_position = batch_positions[key_token];
        if (abs(key_position - query_position) >= Window) { continue; }
        const std::int64_t index =
            kv_batch + d + static_cast<std::int64_t>(D) * (kv_head + KVHeads * key_token);
        const float key_value = __bfloat162float(query_k[index]);
        const float value     = __bfloat162float(query_v[index]);
        const float dot = block_reduce_sum<kSlidingWindowVoltaThreads>(q_value * key_value,
                                                                       warp_sums);
        if (d == 0) { score_s = dot * scale; }
        __syncthreads();
        const float score       = score_s;
        const float next_max    = fmaxf(row_max, score);
        const float old_scale   = row_max == -CUDART_INF_F
                                      ? 0.0f
                                      : exp2_approx((row_max - next_max) * 1.4426950408889634f);
        const float probability = exp2_approx((score - next_max) * 1.4426950408889634f);
        acc     = acc * old_scale + probability * value;
        row_sum = row_sum * old_scale + probability;
        row_max = next_max;
        __syncthreads();
    }

    out[out_index] = __float2bfloat16_rn(row_sum > 0.0f ? acc / row_sum : 0.0f);
}

} // namespace ninfer::ops
