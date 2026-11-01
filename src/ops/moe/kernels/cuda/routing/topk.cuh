#pragma once

#include "ops/common/numeric/math.cuh"
#include "ops/common/cuda/reduce.cuh"

#include <climits>

namespace ninfer::ops::detail::moe_cuda::routing {

struct RankedExpert {
    float score;
    int id;
};

__device__ __forceinline__ bool better(RankedExpert a, RankedExpert b) {
    return a.score > b.score || (a.score == b.score && a.id < b.id);
}

template <int Count>
__device__ __forceinline__ void merge_ranked(RankedExpert (&run)[Count]) {
#pragma unroll
    for (int partner = 1; partner < 32; partner <<= 1) {
        RankedExpert merged[Count];
#pragma unroll
        for (int rank = 0; rank < Count; ++rank) {
            const auto mirror = run[Count - 1 - rank];
            RankedExpert other{__shfl_xor_sync(0xffffffff, mirror.score, partner),
                               __shfl_xor_sync(0xffffffff, mirror.id, partner)};
            merged[rank] = better(run[rank], other) ? run[rank] : other;
        }
#pragma unroll
        for (int stride = Count / 2; stride; stride >>= 1) {
#pragma unroll
            for (int rank = 0; rank < Count; ++rank) {
                if ((rank & stride) == 0 && !better(merged[rank], merged[rank | stride])) {
                    const auto temporary  = merged[rank];
                    merged[rank]          = merged[rank | stride];
                    merged[rank | stride] = temporary;
                }
            }
        }
#pragma unroll
        for (int rank = 0; rank < Count; ++rank) { run[rank] = merged[rank]; }
    }
}

template <class Shape, bool FixedIndices = false>
__device__ __forceinline__ void select_experts(const float* scores, int* ids, float* coefficients,
                                               float* shared, float* logits) {
    constexpr int K = Shape::top_k, E = Shape::experts;
    constexpr int Count = [] {
        int n = 1;
        while (n < K) { n *= 2; }
        return n;
    }();
    constexpr int Local = (E + 31) / 32;
    static_assert(K <= E && K <= 32);
    const int lane = threadIdx.x & 31;
    RankedExpert run[Count];
    if constexpr (Local <= Count) {
#pragma unroll
        for (int item = 0; item < Count; ++item) {
            const int id = lane + item * 32;
            run[item]    = {id < E ? scores[id] : -INFINITY, id < E ? id : INT_MAX};
        }
        if constexpr (FixedIndices) {
            bool ordered = true;
#pragma unroll
            for (int item = 1; item < Count; ++item) {
                ordered &= !better(run[item], run[item - 1]);
            }
            // Fixed indices keep the local ranked run in registers. The uniform fast
            // path preserves already ordered inputs without a data-dependent index.
            if (!__all_sync(0xffffffff, ordered)) {
#pragma unroll
                for (int width = 2; width <= Count; width *= 2) {
#pragma unroll
                    for (int stride = width / 2; stride; stride /= 2) {
#pragma unroll
                        for (int item = 0; item < Count; ++item) {
                            if ((item & stride) == 0) {
                                const int other = item | stride;
                                const auto left = run[item], right = run[other];
                                const bool exchange =
                                    (item & width) == 0 ? better(right, left) : better(left, right);
                                run[item]  = exchange ? right : left;
                                run[other] = exchange ? left : right;
                            }
                        }
                    }
                }
            }
        } else {
#pragma unroll
            for (int i = 1; i < Count; ++i) {
                const auto value = run[i];
                int position     = i;
                while (position > 0 && better(value, run[position - 1])) {
                    run[position] = run[position - 1];
                    --position;
                }
                run[position] = value;
            }
        }
    } else {
#pragma unroll
        for (int rank = 0; rank < Count; ++rank) { run[rank] = {-INFINITY, INT_MAX}; }
#pragma unroll
        for (int item = 0; item < Local; ++item) {
            const int id = lane + item * 32;
            RankedExpert value{id < E ? scores[id] : -INFINITY, id};
#pragma unroll
            for (int rank = 0; rank < Count; ++rank) {
                if (better(value, run[rank])) {
                    const auto previous = run[rank];
                    run[rank]           = value;
                    value               = previous;
                }
            }
        }
    }
    merge_ranked(run);
    if (lane == 0) {
#pragma unroll
        for (int rank = 0; rank < K; ++rank) {
            ids[rank]    = run[rank].id;
            logits[rank] = run[rank].score;
        }
    }
    __syncwarp();
    const float numerator = lane < K ? expf(logits[lane] - logits[0]) : 0;
    float denominator     = common::warp_reduce_sum(numerator);
    denominator           = __shfl_sync(0xffffffff, denominator, 0);
    if (lane < K) { coefficients[lane] = numerator / denominator; }
    if (lane == 0) { *shared = common::sigmoid(scores[E]); }
}

} // namespace ninfer::ops::detail::moe_cuda::routing
