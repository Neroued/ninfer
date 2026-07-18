#pragma once

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

namespace ninfer::ops::detail {

inline constexpr int kSparseMoeExperts = 256;
inline constexpr int kSparseMoeTopK    = 8;

struct SparseMoeRankedValue {
    float value;
    int id;
    int origin;
};

__device__ __forceinline__ bool sparse_moe_ranked_better(const SparseMoeRankedValue& a,
                                                         const SparseMoeRankedValue& b) {
    return a.value > b.value || (a.value == b.value && a.id < b.id);
}

__device__ __forceinline__ SparseMoeRankedValue sparse_moe_warp_best(SparseMoeRankedValue value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        SparseMoeRankedValue other;
        other.value  = __shfl_down_sync(kFullWarpMask, value.value, offset);
        other.id     = __shfl_down_sync(kFullWarpMask, value.id, offset);
        other.origin = __shfl_down_sync(kFullWarpMask, value.origin, offset);
        if (sparse_moe_ranked_better(other, value)) { value = other; }
    }
    value.value  = __shfl_sync(kFullWarpMask, value.value, 0);
    value.id     = __shfl_sync(kFullWarpMask, value.id, 0);
    value.origin = __shfl_sync(kFullWarpMask, value.origin, 0);
    return value;
}

__device__ __forceinline__ void sparse_moe_select_top8_warp(const float* scores, int* ids,
                                                            float* alpha, float* shared_scale,
                                                            float* selected_logits) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    SparseMoeRankedValue local[8];
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int id = lane + item * 32;
        local[item]  = {scores[id], id, lane};
    }
#pragma unroll
    for (int i = 1; i < 8; ++i) {
        const SparseMoeRankedValue value = local[i];
        int position                     = i;
        while (position > 0 && sparse_moe_ranked_better(value, local[position - 1])) {
            local[position] = local[position - 1];
            --position;
        }
        local[position] = value;
    }

    int cursor = 0;
#pragma unroll
    for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
        SparseMoeRankedValue candidate =
            cursor < 8 ? local[cursor] : SparseMoeRankedValue{-CUDART_INF_F, 0x7fffffff, lane};
        const SparseMoeRankedValue winner = sparse_moe_warp_best(candidate);
        if (lane == 0) {
            ids[rank]             = winner.id;
            selected_logits[rank] = winner.value;
        }
        if (lane == winner.origin) { ++cursor; }
        __syncwarp();
    }

    float exponential = 0.0f;
    if (lane < kSparseMoeTopK) { exponential = expf(selected_logits[lane] - selected_logits[0]); }
    float denominator = warp_reduce_sum(exponential);
    denominator       = __shfl_sync(kFullWarpMask, denominator, 0);
    if (lane < kSparseMoeTopK) { alpha[lane] = exponential / denominator; }
    if (lane == 0) { *shared_scale = sigmoid(scores[kSparseMoeExperts]); }
}

} // namespace ninfer::ops::detail
