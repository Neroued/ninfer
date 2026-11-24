#pragma once
#include "ops/softmax_attention/kernels/causal/ragged_grouped.cuh"

#include "ops/softmax_attention/kernels/causal/bf16/tile_io.cuh"
#include "ops/softmax_attention/kernels/causal/bf16/partition.h"
#include "ops/softmax_attention/kernels/causal/partition.h"
#include "ops/softmax_attention/kernels/softmax.cuh"

namespace ninfer::ops::detail {

// Merge one query/head's split statistics once per CTA. Published scalars are separate
// from the reduction/weight storage, so later writes cannot race another warp's scalar read.
template <class Geometry, class Schedule>
__device__ __forceinline__ float
bf16_kv_merge_statistics(const float* partial_m, const float* partial_l, int q_head, int token,
                         int tokens, int splits, float* weights, float* warp_sums, float* scalars) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const auto index   = causal_stat_index<Geometry>(q_head, token, tid, tokens);
    const float m      = tid < splits ? partial_m[index] : -CUDART_INF_F;
    const float warp_m = common::warp_max(m);
    if (lane == 0) warp_sums[warp] = warp_m;
    __syncthreads();
    if (warp == 0) {
        const float maximum =
            common::warp_max(tid < Schedule::kWarps ? warp_sums[tid] : -CUDART_INF_F);
        if (tid == 0) scalars[0] = maximum;
    }
    __syncthreads();
    const float maximum = scalars[0];
    const float l       = tid < splits ? partial_l[index] : 0.0f;
    const float weight =
        l > 0.0f && maximum > -CUDART_INF_F ? common::exp2_approx(m - maximum) : 0.0f;
    const float denominator = common::block_reduce_sum<Schedule::kThreads>(l * weight, warp_sums);
    if (tid == 0) scalars[1] = denominator;
    __syncthreads();
    const float total = scalars[1];
    if (tid < splits) weights[tid] = total > 0.0f ? weight : 0.0f;
    __syncthreads();
    return total;
}

#include "ops/softmax_attention/kernels/causal/bf16/merge.inl"

#define NINFER_CAUSAL_RAGGED
#include "ops/softmax_attention/kernels/causal/bf16/merge.inl"
#undef NINFER_CAUSAL_RAGGED


} // namespace ninfer::ops::detail
