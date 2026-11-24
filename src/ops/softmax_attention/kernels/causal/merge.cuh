#pragma once
#include "ops/softmax_attention/kernels/causal/ragged_grouped.cuh"
#include "core/device.h"
#include "ops/softmax_attention/kernels/causal/operands.h"

#include "ops/softmax_attention/kernels/causal/epilogue.cuh"
#include "ops/softmax_attention/kernels/causal/partition.h"
#include "ops/softmax_attention/kernels/softmax.cuh"

namespace ninfer::ops::detail {

static __global__ void causal_merge_idle_kernel() {}

template <class Geometry>
__device__ __forceinline__ float
causal_merge_natural_statistics(const float* partial_m, const float* partial_l, int q_head,
                                int token, int tokens, int splits, float* weights, float* warp_sums,
                                float* scalars) {
    static_assert(CausalKvPartition::kMaxSplits <= 256);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const auto index   = causal_stat_index<Geometry>(q_head, token, tid, tokens);
    const float m      = tid < splits ? partial_m[index] : -CUDART_INF_F;
    const float warp_m = common::warp_max(m);
    if (lane == 0) warp_sums[warp] = warp_m;
    __syncthreads();
    if (warp == 0) {
        const float maximum = common::warp_max(tid < 8 ? warp_sums[tid] : -CUDART_INF_F);
        if (tid == 0) scalars[0] = maximum;
    }
    __syncthreads();
    const float maximum     = scalars[0];
    const float l           = tid < splits ? partial_l[index] : 0.0f;
    const float weight      = l > 0.0f && maximum > -CUDART_INF_F ? expf(m - maximum) : 0.0f;
    const float denominator = common::block_reduce_sum<256>(l * weight, warp_sums);
    if (tid == 0) scalars[1] = denominator;
    __syncthreads();
    const float total = scalars[1];
    if (tid < splits) weights[tid] = total > 0.0f ? weight : 0.0f;
    __syncthreads();
    return total;
}

#include "ops/softmax_attention/kernels/causal/merge.inl"

#define NINFER_CAUSAL_RAGGED
#include "ops/softmax_attention/kernels/causal/merge.inl"
#undef NINFER_CAUSAL_RAGGED

template <class G, class S, bool MultiBatch, bool Masked, bool InverseRotation, class... Metadata>
void launch_causal_natural_merge(const CausalAttentionOperands& p,
                                 const std::int32_t* valid_columns, CausalKvPartition partition,
                                 CausalPartialView partial, cudaStream_t stream,
                                 Metadata... metadata) {
    static_assert(S::kThreads == 256);
    if (partition.capacity == 1) {
        if constexpr (sizeof...(Metadata) == 0) {
            causal_merge_idle_kernel<<<1, 1, 0, stream>>>();
            CUDA_CHECK(cudaGetLastError());
        }
        return;
    }
    const dim3 grid(G::QHeads, common::div_up(G::kHeadDim, S::kDChunk), p.width * p.batch);
    constexpr auto kernel = [] {
        if constexpr (sizeof...(Metadata) == 0)
            return causal_natural_merge_kernel<G, S::kDChunk, MultiBatch, Masked, InverseRotation>;
        else
            return causal_natural_merge_ragged_kernel<G, S::kDChunk, MultiBatch, Masked,
                                                      InverseRotation>;
    }();
    kernel<<<grid, S::kThreads, 0, stream>>>(partial.acc, partial.maximum, partial.sum, p.positions,
                                             valid_columns, p.width, p.batch, p.visible_capacity,
                                             partition, p.out, metadata...);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
