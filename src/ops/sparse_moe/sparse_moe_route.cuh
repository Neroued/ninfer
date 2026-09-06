#pragma once

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

inline constexpr int kSparseMoeExperts = 256;
inline constexpr int kSparseMoeTopK    = 8;

struct SparseMoeRankedValue {
    float value;
    int id;
};

__device__ __forceinline__ bool sparse_moe_ranked_better(const SparseMoeRankedValue& a,
                                                         const SparseMoeRankedValue& b) {
    return a.value > b.value || (a.value == b.value && a.id < b.id);
}

// Merges the descending run of each lane with the run of its xor partner and keeps the better
// half. The eight exchanges of a step do not depend on each other, so the warp reaches the
// warp-wide top-8 in five merge steps instead of eight dependent reduction rounds. Ranking is a
// total order over distinct expert ids, so the selected set and its order are the same as the
// ones any other correct selection produces.
__device__ __forceinline__ void
sparse_moe_merge_ranked_runs(SparseMoeRankedValue (&run)[kSparseMoeTopK]) {
#pragma unroll
    for (int partner = 1; partner < 32; partner <<= 1) {
        SparseMoeRankedValue merged[kSparseMoeTopK];
#pragma unroll
        for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
            const SparseMoeRankedValue mirror = run[kSparseMoeTopK - 1 - rank];
            SparseMoeRankedValue other;
            other.value  = __shfl_xor_sync(kFullWarpMask, mirror.value, partner);
            other.id     = __shfl_xor_sync(kFullWarpMask, mirror.id, partner);
            merged[rank] = sparse_moe_ranked_better(run[rank], other) ? run[rank] : other;
        }
        // The kept half is bitonic; three compare-exchange stages restore descending order.
#pragma unroll
        for (int stride = kSparseMoeTopK / 2; stride > 0; stride >>= 1) {
#pragma unroll
            for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
                if ((rank & stride) != 0) { continue; }
                const int partner_rank = rank | stride;
                if (!sparse_moe_ranked_better(merged[rank], merged[partner_rank])) {
                    const SparseMoeRankedValue swap = merged[rank];
                    merged[rank]                    = merged[partner_rank];
                    merged[partner_rank]            = swap;
                }
            }
        }
#pragma unroll
        for (int rank = 0; rank < kSparseMoeTopK; ++rank) { run[rank] = merged[rank]; }
    }
}

__device__ __forceinline__ void sparse_moe_select_top8_warp(const float* scores, int* ids,
                                                            float* alpha, float* shared_scale,
                                                            float* selected_logits) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    SparseMoeRankedValue local[kSparseMoeTopK];
#pragma unroll
    for (int item = 0; item < kSparseMoeTopK; ++item) {
        const int id = lane + item * 32;
        local[item]  = {scores[id], id};
    }
#pragma unroll
    for (int i = 1; i < kSparseMoeTopK; ++i) {
        const SparseMoeRankedValue value = local[i];
        int position                     = i;
        while (position > 0 && sparse_moe_ranked_better(value, local[position - 1])) {
            local[position] = local[position - 1];
            --position;
        }
        local[position] = value;
    }
    sparse_moe_merge_ranked_runs(local);

    if (lane == 0) {
#pragma unroll
        for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
            ids[rank]             = local[rank].id;
            selected_logits[rank] = local[rank].value;
        }
    }
    __syncwarp();

    float exponential = 0.0f;
    if (lane < kSparseMoeTopK) { exponential = expf(selected_logits[lane] - selected_logits[0]); }
    float denominator = warp_reduce_sum(exponential);
    denominator       = __shfl_sync(kFullWarpMask, denominator, 0);
    if (lane < kSparseMoeTopK) { alpha[lane] = exponential / denominator; }
    if (lane == 0) { *shared_scale = sigmoid(scores[kSparseMoeExperts]); }
}

} // namespace ninfer::ops::detail
