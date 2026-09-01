#pragma once

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/softmax_attention/dense/packed/kernel.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

namespace ninfer::ops {

inline constexpr int kPackedAttentionVoltaThreads = 128;

__launch_bounds__(kPackedAttentionVoltaThreads, 1) __global__ void
packed_attention_volta_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const std::int32_t* __restrict__ cu_seqlens,
    int segments, int uniform_segment_length, int tokens, __nv_bfloat16* __restrict__ out,
    std::int64_t q_stride_d, std::int64_t q_stride_h, std::int64_t q_stride_t,
    std::int64_t k_stride_d, std::int64_t k_stride_h, std::int64_t k_stride_t,
    std::int64_t v_stride_d, std::int64_t v_stride_h, std::int64_t v_stride_t) {
    constexpr int D = kPackedAttentionHeadDim;
    constexpr float Scale = 0.11785113019775792073f;

    __shared__ float warp_sums[kPackedAttentionVoltaThreads / kWarpSize];
    __shared__ float score_s;
    __shared__ int begin_s;
    __shared__ int end_s;

    const int token = static_cast<int>(blockIdx.x);
    const int head  = static_cast<int>(blockIdx.y);
    const int d     = static_cast<int>(threadIdx.x);
    if (d == 0) {
        if (uniform_segment_length > 0) {
            begin_s = (token / uniform_segment_length) * uniform_segment_length;
            end_s   = min(tokens, begin_s + uniform_segment_length);
        } else {
            begin_s = 0;
            end_s   = 0;
            for (int segment = 0; segment < segments; ++segment) {
                const int begin = cu_seqlens[segment];
                const int end   = cu_seqlens[segment + 1];
                if (token >= begin && token < end) {
                    begin_s = begin;
                    end_s   = end;
                    break;
                }
            }
        }
    }
    __syncthreads();

    const bool live_d = d < D;
    const float q_value = live_d ? __bfloat162float(*packed_attention_ptr(
                                         q, q_stride_d, q_stride_h, q_stride_t, d, head, token))
                                     : 0.0f;
    float row_max = -CUDART_INF_F;
    float row_sum = 0.0f;
    float acc     = 0.0f;
    for (int key_token = begin_s; key_token < end_s; ++key_token) {
        const float key_value = live_d ? __bfloat162float(*packed_attention_ptr(
                                               k, k_stride_d, k_stride_h, k_stride_t, d, head,
                                               key_token))
                                       : 0.0f;
        const float value = live_d ? __bfloat162float(*packed_attention_ptr(
                                           v, v_stride_d, v_stride_h, v_stride_t, d, head,
                                           key_token))
                                   : 0.0f;
        const float dot = block_reduce_sum<kPackedAttentionVoltaThreads>(q_value * key_value,
                                                                         warp_sums);
        if (d == 0) { score_s = dot * Scale; }
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

    if (live_d) {
        out[(static_cast<std::int64_t>(token) * kPackedAttentionHeads + head) * D + d] =
            __float2bfloat16_rn(row_sum > 0.0f ? acc / row_sum : 0.0f);
    }
}

} // namespace ninfer::ops
