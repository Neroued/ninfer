#pragma once

#include "ops/softmax_attention/qsa/indexer/work_map.cuh"
#include "ops/common/cuda/reduce.cuh"

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

namespace ninfer::ops::detail::qsa {

using IndexerScan   = cub::BlockScan<int, 256>;
using IndexerReduce = cub::BlockReduce<std::uint64_t, 256>;

template <int Bits, int Copies>
struct IndexerRadixStorage {
    static constexpr int bins = 1 << Bits;

    union {
        int histogram[Copies][bins];
        IndexerScan::TempStorage scan;
        IndexerReduce::TempStorage reduce;
    } temp;

    std::uint64_t prefix, low, mask;
    int shift, remaining, selected_count;
};

struct IndexerKeyMin {
    __device__ std::uint64_t operator()(std::uint64_t a, std::uint64_t b) const {
        return min(a, b);
    }
};

struct IndexerKeyMax {
    __device__ std::uint64_t operator()(std::uint64_t a, std::uint64_t b) const {
        return max(a, b);
    }
};

struct IndexerStoreKeys {
    std::uint64_t* out;

    __device__ void operator()(int rank, std::uint64_t key) const { out[rank] = key; }
};

struct IndexerStoreIds {
    std::int32_t* out;

    __device__ void operator()(int rank, std::uint64_t key) const {
        out[rank] = common::id_from_order_key(key);
    }
};

// Full radix selection over a unique score/id total order. Prefix refinement stops only
// when exactly 512 keys are >= the prefix. No approximate local candidate truncation.
template <int Items, int Bits, int Copies, class Store>
__device__ void indexer_select_512(std::uint64_t (&keys)[Items], int count,
                                   IndexerRadixStorage<Bits, Copies>& s, Store store) {
    constexpr int Bins = 1 << Bits;
    const int tid      = threadIdx.x;
    if (count <= kIndexerSelected) {
#pragma unroll
        for (int j = 0; j < Items; ++j)
            if (j * 256 + tid < count) store(j * 256 + tid, keys[j]);
        return;
    }
    std::uint64_t lo = ~std::uint64_t(0), hi = 0;
#pragma unroll
    for (int j = 0; j < Items; ++j) {
        if (keys[j]) lo = min(lo, keys[j]);
        hi = max(hi, keys[j]);
    }
    lo = IndexerReduce(s.temp.reduce).Reduce(lo, IndexerKeyMin{});
    if (tid == 0) s.low = lo;
    __syncthreads();
    hi = IndexerReduce(s.temp.reduce).Reduce(hi, IndexerKeyMax{});
    if (tid == 0) {
        s.shift     = ((63 - __clzll(s.low ^ hi)) / Bits) * Bits;
        s.mask      = (Bits == 8 ? s.shift == 56 : s.shift >= 64 - Bits)
                          ? 0
                          : (~std::uint64_t(0) << (s.shift + Bits));
        s.prefix    = hi & s.mask;
        s.remaining = kIndexerSelected;
    }
    __syncthreads();
    for (int shift = s.shift; shift >= 0; shift -= Bits) {
        for (int i = tid; i < Copies * Bins; i += 256) (&s.temp.histogram[0][0])[i] = 0;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < Items; ++j) {
            const auto key = keys[j];
            const int bin =
                key && (key & s.mask) == s.prefix ? int((key >> shift) & (Bins - 1)) : Bins;
            const unsigned peers = __match_any_sync(0xffffffffu, bin);
            if (bin < Bins && (tid % 32) == __ffs(peers) - 1)
                atomicAdd(&s.temp.histogram[Copies == 8 ? tid / 32 : tid / 32 % Copies][bin],
                          __popc(peers));
        }
        __syncthreads();
        int histogram = 0;
#pragma unroll
        for (int copy = 0; copy < Copies; ++copy) {
            if constexpr (Bins == 256)
                histogram += s.temp.histogram[copy][tid];
            else if (tid < Bins)
                histogram += s.temp.histogram[copy][tid];
        }
        __syncthreads();
        int below, total;
        IndexerScan(s.temp.scan).ExclusiveSum(histogram, below, total);
        const int above     = total - below - histogram;
        const int remaining = s.remaining;
        __syncthreads();
        if (above < remaining && above + histogram >= remaining) {
            s.prefix |= std::uint64_t(tid) << shift;
            s.remaining      = remaining - above;
            s.selected_count = histogram;
        }
        __syncthreads();
        if (s.selected_count == s.remaining) break;
        if (tid == 0) s.mask |= std::uint64_t(Bins - 1) << shift;
        __syncthreads();
    }
    const auto threshold = s.prefix;
    int kept             = 0;
#pragma unroll
    for (int j = 0; j < Items; ++j) kept += keys[j] >= threshold;
    int rank;
    IndexerScan(s.temp.scan).ExclusiveSum(kept, rank);
#pragma unroll
    for (int j = 0; j < Items; ++j)
        if (keys[j] >= threshold) store(rank++, keys[j]);
}

template <int Partition, int Bits = 8, int Copies = 8, int MinBlocks = 1, int MaxQueries = 64,
          bool Reverse = false>
__global__ __launch_bounds__(256, MinBlocks) void indexer_partition_kernel(
    const __grid_constant__ IndexerArguments a) {
    if (a.columns <= kIndexerSelected) return;
    __shared__ IndexerJobs<MaxQueries> jobs;
    __shared__ IndexerRadixStorage<Bits, Copies> radix;
    if (!jobs.build<1, Partition>(a)) return;
    const int total = jobs.prefix[jobs.groups];
    for (int step = blockIdx.x; step < total; step += gridDim.x) {
        const int job  = Reverse ? total - 1 - step : step;
        const int slot = jobs.group_for(job), part = job - jobs.prefix[slot];
        const int first = part * Partition;
        const int count = min(Partition, jobs.visible[slot] - first);
        std::uint64_t keys[Partition / 256];
#pragma unroll
        for (int j = 0; j < Partition / 256; ++j) {
            const int i = j * 256 + threadIdx.x;
            keys[j]     = i < count
                              ? common::score_id_order_key(
                                a.scores[std::int64_t(slot) * a.columns + first + i], first + i)
                              : 0;
        }
        // A discarded key already has 512 better keys in its own partition, so it cannot
        // belong to the global Top512. Only the final partition can emit fewer than 512.
        auto* out = a.candidates + (std::int64_t(slot) * a.partitions + part) * kIndexerSelected;
        indexer_select_512(keys, count, radix, IndexerStoreKeys{out});
        __syncthreads();
    }
}

template <int Partition, int Bits = 8, int Copies = 8,
          int MaxPartitions = kIndexerMaxBlocks / Partition>
__global__
__launch_bounds__(256) void indexer_finish_kernel(const __grid_constant__ IndexerArguments a) {
    __shared__ IndexerRadixStorage<Bits, Copies> radix;
    const int slot = blockIdx.x, tid = threadIdx.x, visible = a.visible(slot);
    const auto column = a.column(slot);
    auto* out         = a.ids + column * kIndexerSelected;
    if (tid == 0) a.counts[column] = min(visible, kIndexerSelected);
    if (visible <= kIndexerSelected) {
        for (int i = tid; i < visible; i += 256) out[i] = i;
        return;
    }
    const int parts = (visible + Partition - 1) / Partition;
    const int count =
        (parts - 1) * kIndexerSelected + min(kIndexerSelected, visible - (parts - 1) * Partition);
    const auto* input   = a.candidates + std::int64_t(slot) * a.partitions * kIndexerSelected;
    constexpr int Items = MaxPartitions * kIndexerSelected / 256;
    std::uint64_t keys[Items];
#pragma unroll
    for (int j = 0; j < Items; ++j) {
        const int i = j * 256 + tid;
        keys[j]     = i < count ? input[i] : 0;
    }
    indexer_select_512(keys, count, radix, IndexerStoreIds{out});
}
} // namespace ninfer::ops::detail::qsa
