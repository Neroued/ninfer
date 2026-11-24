#pragma once

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

struct Bf16KvLivePartition {
    int splits;
    int keys_per_split;
};

// Capture fixes capacity; producer and merge derive the same partition from
// the live row length, including ragged batches and Graph replay.
struct Bf16KvPartition {
    int capacity      = 1;
    int normal_target = 1;
    int long_target   = 1;
    int key_rows      = 32;

    __host__ __device__ static int ceil_div(int n, int d) { return (n + d - 1) / d; }

    __host__ __device__ Bf16KvLivePartition live(int visible) const {
        const int target  = visible >= 32768 ? long_target : normal_target;
        const int rounded = ceil_div(ceil_div(visible, target), key_rows) * key_rows;
        const int keys    = rounded < 64 ? 64 : rounded;
        return {ceil_div(visible, keys), keys};
    }

    // Between key-tile rounding points, the split count grows with length. A
    // complete target*chunk boundary reaches target; otherwise high is maximal.
    int interval_capacity(int low, int high, int target) const {
        if (low > high) return 0;
        const int boundary = high / (target * key_rows) * (target * key_rows);
        if (boundary >= low && boundary >= target * 64) return target;
        return live(high).splits;
    }
};

} // namespace ninfer::ops::detail
