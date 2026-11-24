#pragma once

#include "ops/softmax_attention/kernels/causal/partition.h"
#include "ops/softmax_attention/kernels/causal/ragged.h"

namespace ninfer::ops::detail {

__device__ __forceinline__ int causal_ragged_window(const CausalShortMetadata& metadata,
                                                    const int* positions, int row,
                                                    int logical_capacity) {
    if (row >= metadata.segments.count) return 0;
    const int segment = metadata.segments.entry(row);
    const int begin   = metadata.cu_seqlens[segment];
    const int last    = positions[begin + metadata.seq_lengths[segment] - 1];
    return positions[begin] < 0 || last < 0 || last >= logical_capacity ? 0 : last + 1;
}

template <class Partition>
__device__ __forceinline__ CausalSplitPool causal_ragged_pool(const Partition& partition,
                                                              const CausalShortMetadata& metadata,
                                                              const int* positions) {
    const int lane = threadIdx.x & 31;
    return causal_split_pool_windows(
        partition, metadata.segments.count,
        causal_ragged_window(metadata, positions, lane, metadata.visible_capacity),
        causal_ragged_window(metadata, positions, lane + 32, metadata.visible_capacity));
}

template <bool Direct, class Partition>
__device__ __forceinline__ CausalProducerSplit causal_ragged_producer(
    const Partition& partition, const CausalShortMetadata& metadata, const int* positions) {
    const int row = blockIdx.z, split = blockIdx.y;
    const int offset = row * partition.capacity;
    if constexpr (!Direct) {
        if (use_causal_split_pool(metadata.segments.count, partition.capacity))
            return causal_ragged_pool(partition, metadata, positions).worker(offset + split);
    }
    return {row, split, {offset, -1, 0}};
}

template <class Partition>
__device__ __forceinline__ CausalRowSplits causal_ragged_merge(const Partition& partition,
                                                               const CausalShortMetadata& metadata,
                                                               const int* positions, int row) {
    if (use_causal_split_pool(metadata.segments.count, partition.capacity))
        return causal_ragged_pool(partition, metadata, positions).row(row);
    return {row * partition.capacity, -1,
            causal_ragged_window(metadata, positions, row, metadata.visible_capacity)};
}

} // namespace ninfer::ops::detail
