#pragma once

#include <cuda_runtime.h>
#include <climits>

namespace ninfer::ops::detail::moe_cuda::block_scaled {

template <int Experts, int TopK, class Group>
__device__ __forceinline__ bool fully_covered_groups(Group g, int assignments, bool density,
                                                     bool ordered) {
    if (!ordered) return false;
    if (!density) {
        // Distinct route IDs give at most one assignment per expert and token.
        // Equal nonempty counts at T therefore cover every token in each group.
        const int tokens = assignments / TopK;
        return g.load_range && g.load_range[0] == tokens && g.load_range[1] == tokens;
    }
    // Distinct IDs bound every expert's count by T. Reaching active*T means
    // every active expert covers every token, independent of route-slot order.
    const int active = g.jobs[2];
    return active > 0 && g.offsets[Experts] == active * (assignments / TopK);
}

// Publish [minimum nonempty load, maximum load, active experts].
// An empty selection produces [INT_MAX, 0, 0].
template <int Experts>
__device__ __forceinline__ void publish_load_range(int* range, int count) {
    if (!range) return;
    constexpr int warps = (Experts + 31) / 32;
    __shared__ int minima[warps], maxima[warps], actives[warps];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int low = count ? count : INT_MAX, high = count;
    int active = __popc(__ballot_sync(0xffffffff, count > 0));
#pragma unroll
    for (int d = 16; d; d /= 2) {
        low  = min(low, __shfl_down_sync(0xffffffff, low, d));
        high = max(high, __shfl_down_sync(0xffffffff, high, d));
    }
    if (!lane) {
        minima[warp]  = low;
        maxima[warp]  = high;
        actives[warp] = active;
    }
    __syncthreads();
    if (!warp) {
        low    = lane < warps ? minima[lane] : INT_MAX;
        high   = lane < warps ? maxima[lane] : 0;
        active = lane < warps ? actives[lane] : 0;
#pragma unroll
        for (int d = 16; d; d /= 2) {
            low  = min(low, __shfl_down_sync(0xffffffff, low, d));
            high = max(high, __shfl_down_sync(0xffffffff, high, d));
            active += __shfl_down_sync(0xffffffff, active, d);
        }
        if (!lane) {
            range[0] = low;
            range[1] = high;
            range[2] = active;
        }
    }
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled
