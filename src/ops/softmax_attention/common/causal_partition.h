#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

// Capture reserves partials for the largest live row. Producer and merge use
// the same live count; a wider capture never changes a row's work partition.
struct CausalKvPartition {
    static constexpr int kMaxSplits = 256;
    int capacity                    = 1;
    int target                      = 1;
    int key_shift                   = 6; // log2 of the minimum KV keys per split

    __host__ __device__ int active(int visible) const {
        const int count = (visible + (1 << key_shift) - 1) >> key_shift;
        return count < target ? count : target;
    }
};

// Complete groups of independent query tiles should fill at least 90% of the SMs.
// Each dtype chooses whether a shortfall warrants a larger CTA budget.
inline constexpr bool causal_query_tiles_underfill_sms(int independent_tiles,
                                                       int multiprocessor_count) {
    const std::int64_t filled =
        (multiprocessor_count / independent_tiles) * static_cast<std::int64_t>(independent_tiles);
    return filled < static_cast<std::int64_t>(multiprocessor_count) * 9 / 10;
}

inline constexpr int causal_partition_target(std::int64_t cta_budget, int independent_tiles) {
    return static_cast<int>(
        std::clamp<std::int64_t>(cta_budget / independent_tiles, 1, CausalKvPartition::kMaxSplits));
}
inline constexpr int kCausalTiledMaxSplits = 8;

// Minimize waves per KV partition, retaining fewer partitions on a tie.
// The bound limits FP32 partial traffic; live rows cap the count at
// ceil(visible_keys / 512). Count changes work and storage, never kernel topology.
inline int causal_tiled_split_target(std::int64_t ctas, int multiprocessor_count) {
    int selected = 1;
    auto waves   = (ctas + multiprocessor_count - 1) / multiprocessor_count;
    for (int splits = 2; splits <= kCausalTiledMaxSplits; ++splits) {
        const auto next = (ctas * splits + multiprocessor_count - 1) / multiprocessor_count;
        if (next * selected < waves * splits) {
            selected = splits;
            waves    = next;
        }
    }
    return selected;
}

inline CausalKvPartition make_causal_tiled_partition(std::int64_t ctas, int visible_capacity,
                                                     int multiprocessor_count) {
    CausalKvPartition partition{1, causal_tiled_split_target(ctas, multiprocessor_count), 9};
    partition.capacity = partition.active(visible_capacity);
    return partition;
}


} // namespace ninfer::ops::detail
