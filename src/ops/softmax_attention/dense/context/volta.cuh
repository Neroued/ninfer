#pragma once

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/softmax_attention/common/context_query.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

namespace ninfer::ops {

inline constexpr int kContextAttentionVoltaThreads = kContextQueryHeadDim;

__launch_bounds__(kContextAttentionVoltaThreads, 1) __global__ void
context_attention_volta_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ query_k,
    const __nv_bfloat16* __restrict__ query_v,
    const std::int32_t* __restrict__ context_lengths,
    const std::int32_t* __restrict__ valid_columns,
    const std::int32_t* __restrict__ table_rows,
    const __nv_bfloat16* __restrict__ context_k,
    const __nv_bfloat16* __restrict__ context_v,
    const std::int32_t* __restrict__ block_tables, int physical_pages, int logical_pages,
    int tokens, float scale,
    __nv_bfloat16* __restrict__ out) {
    constexpr int D       = kContextQueryHeadDim;
    constexpr int QHeads  = kContextQueryQHeads;
    constexpr int KVHeads = kContextQueryKVHeads;

    __shared__ float warp_sums[kContextAttentionVoltaThreads / kWarpSize];
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

    const int kv_head = q_head / (QHeads / KVHeads);
    const float q_value = __bfloat162float(q[out_index]);
    const int length = context_lengths[batch];
    const auto* block_table =
        block_tables + static_cast<std::int64_t>(table_rows[batch]) * logical_pages;
    float row_max = -CUDART_INF_F;
    float row_sum = 0.0f;
    float acc     = 0.0f;

    for (int key = 0; key < length; ++key) {
        const int physical_page = paged_kv_physical_page(block_table, key);
        const std::int64_t index = static_cast<std::int64_t>(d) + static_cast<std::int64_t>(D) *
            ((key & kPagedKVPageMask) + kPagedKVPageSize *
                (physical_page + static_cast<std::int64_t>(physical_pages) * kv_head));
        const float key_value = __bfloat162float(context_k[index]);
        const float value     = __bfloat162float(context_v[index]);
        const float dot = block_reduce_sum<kContextAttentionVoltaThreads>(q_value * key_value,
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

    for (int key_token = 0; key_token < valid; ++key_token) {
        const std::int64_t index =
            kv_batch + d + static_cast<std::int64_t>(D) * (kv_head + KVHeads * key_token);
        const float key_value = __bfloat162float(query_k[index]);
        const float value     = __bfloat162float(query_v[index]);
        const float dot = block_reduce_sum<kContextAttentionVoltaThreads>(q_value * key_value,
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
