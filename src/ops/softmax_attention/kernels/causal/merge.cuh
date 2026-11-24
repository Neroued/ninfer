#pragma once
#include "core/device.h"
#include "ops/softmax_attention/kernels/causal/operands.h"

#include "ops/softmax_attention/kernels/causal/epilogue.cuh"
#include "ops/softmax_attention/kernels/causal/partition.h"
#include "ops/softmax_attention/kernels/softmax.cuh"

namespace ninfer::ops::detail {

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

template <class Geometry, int DChunk, bool MultiBatch, bool Masked, bool InverseRotation>
__launch_bounds__(256) __global__
    void causal_natural_merge_kernel(const float* partial_acc, const float* partial_m,
                                     const float* partial_l, const std::int32_t* positions,
                                     const std::int32_t* valid_columns, std::int32_t tokens,
                                     std::int32_t batch_size, CausalKvPartition partition,
                                     __nv_bfloat16* out) {
    static_assert(Geometry::kHeadDim == kCausalHeadDim);
    static_assert(DChunk > 0 && DChunk <= kCausalHeadDim);
    static_assert(!InverseRotation || DChunk == kCausalHeadDim);
    const int q_head      = static_cast<int>(blockIdx.x);
    const int d_start     = static_cast<int>(blockIdx.y) * DChunk;
    const int flat_column = static_cast<int>(blockIdx.z);
    int batch             = 0;
    int token             = flat_column;
    if constexpr (MultiBatch) {
        batch = flat_column / tokens;
        token = flat_column - batch * tokens;
    }
    const int tid         = static_cast<int>(threadIdx.x);
    const int split_count = partition.capacity;
    if (q_head >= Geometry::QHeads || token >= tokens) return;
    if constexpr (MultiBatch) {
        if (batch >= batch_size) return;
    }
    if constexpr (MultiBatch) positions += static_cast<std::int64_t>(batch) * tokens;
    const int window  = positions[tokens - 1] + 1;
    int output_column = token;
    if constexpr (MultiBatch) output_column += batch * tokens;
    if constexpr (Masked) {
        const int absolute_column = token;
        if (absolute_column >= valid_columns[batch]) {
            if (tid < DChunk && d_start + tid < 256)
                out[causal_q_index<Geometry>(q_head, d_start + tid, output_column)] =
                    __float2bfloat16(0.0f);
            return;
        }
    }

    if constexpr (MultiBatch) {
        partial_acc +=
            static_cast<std::int64_t>(batch) * 256 * Geometry::QHeads * tokens * split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    }
    const int active_splits = partition.active(window);
    __shared__ float weights[256], warp_sums[8], scalars[2];
    const float head_l = causal_merge_natural_statistics<Geometry>(
        partial_m, partial_l, q_head, token, tokens, active_splits, weights, warp_sums, scalars);
    const int d = d_start + tid;
    if (tid >= DChunk || d >= 256) return;
    float numerator = 0.0f;
    for (int split = 0; split < active_splits; ++split) {
        if (weights[split] != 0.0f)
            numerator +=
                partial_acc[causal_partial_index<Geometry>(q_head, d, token, split, tokens)] *
                weights[split];
    }

    const float value = head_l > 0.0F ? numerator / head_l : 0.0F;
    if constexpr (InverseRotation) {
        __shared__ float normalized[kCausalHeadDim];
        normalized[tid] = value;
        __syncthreads();
        if (tid < 32)
            causal_store_inverse_rotated_row<Geometry>(normalized, out, q_head, output_column);
    } else {
        causal_store_output(out + causal_q_index<Geometry>(q_head, d, output_column), value);
    }
}

template <class G, class S, bool MultiBatch, bool Masked, bool InverseRotation>
void launch_causal_natural_merge(const CausalAttentionOperands& p,
                                 const std::int32_t* valid_columns, CausalKvPartition partition,
                                 CausalPartialView partial, cudaStream_t stream) {
    static_assert(S::kThreads == 256);
    const dim3 grid(G::QHeads, common::div_up(G::kHeadDim, S::kDChunk), p.width * p.batch);
    causal_natural_merge_kernel<G, S::kDChunk, MultiBatch, Masked, InverseRotation>
        <<<grid, S::kThreads, 0, stream>>>(partial.acc, partial.maximum, partial.sum, p.positions,
                                           valid_columns, p.width, p.batch, partition, p.out);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
