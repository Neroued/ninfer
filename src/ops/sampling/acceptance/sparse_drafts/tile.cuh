#pragma once

#include "ops/sampling/acceptance/kernel.cuh"
#include "ops/sampling/kernels/selection.cuh"

namespace ninfer::ops::sparse_drafts_detail {
using namespace sampling_detail;

template <class Key>
struct BestPair {
    Key first, second;

    __device__ void add(Key key) {
        second = max(second, min(first, key));
        first  = max(first, key);
    }
};

template <class Key>
__device__ Key warp_max(Key value) {
    if constexpr (sizeof(Key) == 4)
        return __reduce_max_sync(0xffffffffU, value);
    else {
        // Select the score bits first, then break ties on the token ID.
        const auto high = __reduce_max_sync(0xffffffffU, static_cast<unsigned>(value >> 32));
        const auto low =
            __reduce_max_sync(0xffffffffU, value >> 32 == high ? static_cast<unsigned>(value) : 0U);
        return (Key(high) << 32) | low;
    }
}

template <int Threads, class Key>
__device__ BestPair<Key> block_best_pair(BestPair<Key> pair, BestPair<Key>* warp_pairs) {
    const int tid    = threadIdx.x;
    const Key first  = warp_max(pair.first);
    const Key second = warp_max(pair.first == first ? pair.second : pair.first);
    if ((tid & 31) == 0) warp_pairs[tid / 32] = {first, second};
    __syncthreads();
    if (tid < 32) {
        pair           = tid < Threads / 32 ? warp_pairs[tid] : BestPair<Key>{};
        const Key best = warp_max(pair.first);
        return {best, warp_max(pair.first == best ? pair.second : pair.first)};
    }
    return {};
}

} // namespace ninfer::ops::sparse_drafts_detail
