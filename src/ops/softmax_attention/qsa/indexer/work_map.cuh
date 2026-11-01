#pragma once

#include "ops/softmax_attention/qsa/indexer/arguments.h"

namespace ninfer::ops::detail::qsa {
template <int Capacity = 64>
struct IndexerJobs {
    int visible[Capacity];
    int prefix[Capacity + 1];
    int groups;

    __device__ int tile_visible(const IndexerArguments& a, int b, int first, int tile) const {
        // Live queries are a prefix with consecutive positions, so its last query has the
        // largest causal history. No position from an inert suffix is inspected.
        const int end = min(min(a.width, a.valid[b] - a.begin), first + tile);
        return end > first ? visible[b * a.width + end - 1] : 0;
    }

    // A query tile never crosses batch rows. Every CTA rebuilds this small immutable map,
    // so actual context lengths distribute work without a Host read or a global queue.
    template <int QueryTile, int KeyTile>
    __device__ bool build(const IndexerArguments& a) {
        const int tid    = threadIdx.x;
        const int length = tid < a.width * a.batch ? a.visible(tid) : 0;
        if (tid < a.width * a.batch) visible[tid] = length;
        if (!__syncthreads_or(length > kIndexerSelected)) return false;
        if (tid < 32) {
            const int per_row = (a.width + QueryTile - 1) / QueryTile;
            const int count   = per_row * a.batch;
            int carry         = 0;
#pragma unroll
            for (int half = 0; half < Capacity / 32; ++half) {
                const int g = tid + half * 32;
                const int n =
                    g < count ? tile_visible(a, g / per_row, (g % per_row) * QueryTile, QueryTile)
                              : 0;
                int sum = n > kIndexerSelected ? (n + KeyTile - 1) / KeyTile : 0;
#pragma unroll
                for (int offset = 1; offset < 32; offset *= 2) {
                    const int previous = __shfl_up_sync(0xffffffffu, sum, offset);
                    if (tid >= offset) sum += previous;
                }
                prefix[g + 1] = carry + sum;
                carry += __shfl_sync(0xffffffffu, sum, 31);
            }
            if (tid == 0) {
                groups    = count;
                prefix[0] = 0;
            }
        }
        __syncthreads();
        return true;
    }

    __device__ int group_for(int job) const {
        int lo = 0, hi = groups;
        while (lo + 1 < hi) {
            const int mid = (lo + hi) / 2;
            if (prefix[mid] <= job)
                lo = mid;
            else
                hi = mid;
        }
        return lo;
    }
};

} // namespace ninfer::ops::detail::qsa
