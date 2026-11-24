#pragma once

#include "ops/softmax_attention/kernels/causal/bf16/tile_io.cuh"
#include "ops/softmax_attention/kernels/causal/bf16/partition.h"
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

template <typename Geometry, class Schedule, bool MultiBatch, bool Masked>
__launch_bounds__(Schedule::kThreads) __global__
    void bf16_kv_merge_kernel(const float* partial_acc, const float* partial_m,
                              const float* partial_l, const std::int32_t* positions,
                              const std::int32_t* valid_columns, std::int32_t tokens,
                              std::int32_t batch_size, Bf16KvPartition partition,
                              __nv_bfloat16* out) {
    const int split_count = partition.capacity;
    constexpr int DChunk  = Schedule::kDChunk;
    static_assert(DChunk <= Geometry::kHeadDim);

    const int q_head      = static_cast<int>(blockIdx.x);
    const int d_start     = static_cast<int>(blockIdx.y) * DChunk;
    const int flat_column = static_cast<int>(blockIdx.z);
    int batch             = 0;
    int token             = flat_column;
    if constexpr (MultiBatch) {
        batch = flat_column / tokens;
        token = flat_column - batch * tokens;
    }
    const int tid = threadIdx.x;
    if (q_head >= Geometry::QHeads || token >= tokens) { return; }
    if constexpr (MultiBatch) {
        if (batch >= batch_size) { return; }
    }

    if constexpr (MultiBatch) { positions += batch * tokens; }
    const int live_columns = Masked ? valid_columns[batch] : tokens;
    int output_column      = token;
    if constexpr (MultiBatch) { output_column += batch * tokens; }
    if constexpr (Masked) {
        const int absolute_column = token;
        if (absolute_column >= live_columns) {
            if (tid < DChunk && d_start + tid < Geometry::kHeadDim)
                out[causal_q_index<Geometry>(q_head, d_start + tid, output_column)] =
                    __float2bfloat16(0.0f);
            return;
        }
    }

    if constexpr (MultiBatch) {
        const std::int64_t partial_acc_row = static_cast<std::int64_t>(batch) * Geometry::kHeadDim *
                                             Geometry::QHeads * tokens * split_count;
        const std::int64_t partial_stat_row =
            static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_acc += partial_acc_row;
        partial_m += partial_stat_row;
        partial_l += partial_stat_row;
    }

    const int window             = positions[live_columns - 1] + 1;
    const int active_split_count = partition.live(window).splits;

    __shared__ float weights[Schedule::kThreads], warp_sums[Schedule::kWarps], scalars[2];
    const float head_l = bf16_kv_merge_statistics<Geometry, Schedule>(
        partial_m, partial_l, q_head, token, tokens, active_split_count, weights, warp_sums,
        scalars);
    const int d = d_start + tid;
    if (tid >= DChunk || d >= Geometry::kHeadDim) return;
    float numerator = 0.0f;
    for (int split = 0; split < active_split_count; ++split) {
        if (weights[split] != 0.0f)
            numerator +=
                partial_acc[causal_partial_index<Geometry>(q_head, d, token, split, tokens)] *
                weights[split];
    }

    const float value = (head_l > 0.0f) ? numerator / head_l : 0.0f;
    out[causal_q_index<Geometry>(q_head, d, output_column)] = __float2bfloat16(value);
}

} // namespace ninfer::ops::detail
