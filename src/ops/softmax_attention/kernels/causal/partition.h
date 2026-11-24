#pragma once

#include "ninfer/ops/context_bucket.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

// Launch capacity and split targets are fixed at capture. The pool can assign a
// row more splits than its static share when other rows need fewer.
struct CausalKvPartition {
    static constexpr int kMaxSplits = 256;
    int capacity                    = 1;
    int target                      = 1;
    int key_shift                   = 6; // log2 of the minimum KV keys per split
    int single_target               = 1;

    __host__ __device__ int active(int visible) const { return live_splits(visible, target); }

    __device__ int single_live_splits(int visible) const {
        return live_splits(visible, single_target);
    }

private:
    __host__ __device__ int live_splits(int visible, int limit) const {
        const int count = (visible + (1 << key_shift) - 1) >> key_shift;
        return count < limit ? count : limit;
    }
};

inline constexpr int kCausalTiledMaxSplits = 8;

// Minimize waves per KV partition, retaining fewer partitions on a tie.
// The bound limits FP32 partial traffic; live rows cap the count at
// ceil(visible_keys / 512). Count changes work and storage, never kernel topology.
inline CausalKvPartition select_causal_tiled_partition(int heads, std::int64_t tiles,
                                                       int visible_capacity,
                                                       int multiprocessor_count,
                                                       int split_limit = kCausalTiledMaxSplits) {
    const std::int64_t ctas = heads * tiles;
    int selected            = 1;
    auto waves              = (ctas + multiprocessor_count - 1) / multiprocessor_count;
    for (int splits = 2; splits <= split_limit; ++splits) {
        const auto next = (ctas * splits + multiprocessor_count - 1) / multiprocessor_count;
        if (next * selected < waves * splits) {
            selected = splits;
            waves    = next;
        }
    }
    CausalKvPartition partition{1, selected, 9};
    partition.capacity = partition.active(visible_capacity);
    return partition;
}

inline constexpr int causal_partition_target(std::int64_t cta_budget, int independent_tiles) {
    return static_cast<int>(
        std::clamp<std::int64_t>(cta_budget / independent_tiles, 1, CausalKvPartition::kMaxSplits));
}

// Complete groups of independent query tiles should fill at least 90% of the SMs.
inline constexpr bool causal_query_tiles_underfill_sms(int independent_tiles, int sms) {
    const auto filled = (sms / independent_tiles) * static_cast<std::int64_t>(independent_tiles);
    return filled < static_cast<std::int64_t>(sms) * 9 / 10;
}

inline constexpr int kMaxPooledBatch = 64;

// Each active row takes a slot before redistribution. The maximum numerator is
// (64 * 256 - 1) * 262144, which fits in uint32_t.
static_assert(std::uint64_t(kMaxPooledBatch) * CausalKvPartition::kMaxSplits *
                  context_bucket_tokens(ContextBucket::K256) <=
              (std::uint64_t{1} << 32));

#if defined(__CUDACC__)

struct CausalRowSplits {
    int offset;
    int count; // -1 selects the static partition, 0 denotes an inactive pooled row.
    int window;
};

struct CausalProducerSplit {
    int batch;
    int split;
    CausalRowSplits slots;
};

inline constexpr unsigned kCausalPoolWarp = 0xffffffffU;

struct CausalSplitPool {
    CausalRowSplits low, high;

    __device__ __forceinline__ CausalRowSplits row(int batch) const {
        const auto slots = batch < 32 ? low : high;
        const int lane   = batch & 31;
        return {__shfl_sync(kCausalPoolWarp, slots.offset, lane),
                __shfl_sync(kCausalPoolWarp, slots.count, lane),
                __shfl_sync(kCausalPoolWarp, slots.window, lane)};
    }

    __device__ __forceinline__ CausalProducerSplit worker(int slot) const {
        const unsigned lo =
            __ballot_sync(kCausalPoolWarp, low.offset <= slot && slot < low.offset + low.count);
        int batch = __ffs(lo) - 1;
        if (lo == 0) {
            const unsigned hi = __ballot_sync(kCausalPoolWarp, high.offset <= slot &&
                                                                   slot < high.offset + high.count);
            if (hi == 0) return {-1, 0, {}};
            batch = __ffs(hi) - 1 + 32;
        }
        const auto slots = row(batch);
        return {batch, slot - slots.offset, slots};
    }
};

template <bool Masked>
__device__ __forceinline__ int causal_pool_window(const int* positions, const int* valid_columns,
                                                  int width, int batch, int batch_size,
                                                  int logical_capacity) {
    if (batch >= batch_size) return 0;
    const int live = Masked ? valid_columns[batch] : width;
    if (live <= 0) return 0;
    const int first = positions[batch * width];
    const int last  = positions[batch * width + live - 1];
    return first < 0 || last < 0 || (logical_capacity > 0 && last >= logical_capacity) ? 0
                                                                                       : last + 1;
}

// Every full warp computes the same allocation. The two halves keep batch order
// in the greedy pass and in the exclusive scan. Producers and merges both use
// these offsets into the original batch * capacity partial slots.
template <class Partition>
__device__ __forceinline__ CausalSplitPool causal_split_pool_windows(const Partition& partition,
                                                                     int batch_size, int w0,
                                                                     int w1) {
    const int lane  = threadIdx.x & 31;
    const int first = __shfl_sync(kCausalPoolWarp, w0, 0);
    if (__all_sync(kCausalPoolWarp, (lane >= batch_size || w0 == first) &&
                                        (lane + 32 >= batch_size || w1 == first))) {
        // Equal demand gives every row the same share in the first pass.
        const int count = first ? min(partition.capacity, partition.single_live_splits(first)) : 0;
        return {{lane * count, lane < batch_size ? count : 0, w0},
                {(lane + 32) * count, lane + 32 < batch_size ? count : 0, w1}};
    }

    const int cap0 = w0 ? partition.single_live_splits(w0) : 0;
    const int cap1 = w1 ? partition.single_live_splits(w1) : 0;
    int count0 = cap0 > 0, count1 = cap1 > 0;
    int remaining =
        batch_size * partition.capacity - __reduce_add_sync(kCausalPoolWarp, count0 + count1);
    for (int pass = 0; pass < 2 && remaining > 0; ++pass) {
        const int room0 = cap0 - count0, room1 = cap1 - count1;
        const unsigned weight0 = room0 > 0 ? w0 : 0;
        const unsigned weight1 = room1 > 0 ? w1 : 0;
        const unsigned total   = __reduce_add_sync(kCausalPoolWarp, weight0 + weight1);
        if (total == 0) break;
        const int add0 = min(room0, int(unsigned(remaining) * weight0 / total));
        const int add1 = min(room1, int(unsigned(remaining) * weight1 / total));
        count0 += add0;
        count1 += add1;
        remaining -= __reduce_add_sync(kCausalPoolWarp, add0 + add1);
    }
    if (remaining > 0) {
        const unsigned lo     = __ballot_sync(kCausalPoolWarp, count0 < cap0);
        const unsigned hi     = __ballot_sync(kCausalPoolWarp, count1 < cap1);
        const unsigned prefix = (1U << lane) - 1U;
        count0 += count0 < cap0 && __popc(lo & prefix) < remaining;
        count1 += count1 < cap1 && __popc(lo) + __popc(hi & prefix) < remaining;
    }
    int prefix0 = count0, prefix1 = count1;
#    pragma unroll
    for (int step = 1; step < 32; step *= 2) {
        const int prev0 = __shfl_up_sync(kCausalPoolWarp, prefix0, step);
        const int prev1 = __shfl_up_sync(kCausalPoolWarp, prefix1, step);
        if (lane >= step) {
            prefix0 += prev0;
            prefix1 += prev1;
        }
    }
    const int low_slots = __shfl_sync(kCausalPoolWarp, prefix0, 31);
    return {{prefix0 - count0, count0, w0}, {low_slots + prefix1 - count1, count1, w1}};
}

template <bool Masked, class Partition>
__device__ __forceinline__ CausalSplitPool causal_split_pool(const Partition& partition,
                                                             const int* positions,
                                                             const int* valid_columns, int width,
                                                             int batch_size, int logical_capacity) {
    const int lane = threadIdx.x & 31;
    const int w0   = causal_pool_window<Masked>(positions, valid_columns, width, lane, batch_size,
                                                logical_capacity);
    const int w1   = causal_pool_window<Masked>(positions, valid_columns, width, lane + 32,
                                                batch_size, logical_capacity);
    return causal_split_pool_windows(partition, batch_size, w0, w1);
}

__device__ __forceinline__ bool use_causal_split_pool(int batch, int capacity) {
    return batch > 1 && batch <= kMaxPooledBatch && capacity > 1;
}

template <bool MultiBatch, bool Masked, bool Direct, class Partition>
__device__ __forceinline__ CausalProducerSplit causal_producer_split(const Partition& partition,
                                                                     const int* positions,
                                                                     const int* valid_columns,
                                                                     int width,
                                                                     int logical_capacity = 0) {
    const int batch  = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split  = static_cast<int>(blockIdx.y);
    const int offset = batch * partition.capacity;
    if constexpr (MultiBatch && !Direct) {
        const int batch_size = static_cast<int>(gridDim.z);
        if (use_causal_split_pool(batch_size, partition.capacity)) {
            return causal_split_pool<Masked>(partition, positions, valid_columns, width, batch_size,
                                             logical_capacity)
                .worker(offset + split);
        }
    }
    return {batch, split, {offset, -1, 0}};
}

template <bool MultiBatch, bool Masked, class Partition>
__device__ __forceinline__ CausalRowSplits causal_merge_splits(const Partition& partition,
                                                               const int* positions,
                                                               const int* valid_columns, int width,
                                                               int live, int batch, int batch_size,
                                                               int logical_capacity = 0) {
    if constexpr (MultiBatch) {
        if (use_causal_split_pool(batch_size, partition.capacity)) {
            return causal_split_pool<Masked>(partition, positions, valid_columns, width, batch_size,
                                             logical_capacity)
                .row(batch);
        }
    }
    return {batch * partition.capacity, -1, positions[batch * width + live - 1] + 1};
}

#endif // __CUDACC__

} // namespace ninfer::ops::detail
