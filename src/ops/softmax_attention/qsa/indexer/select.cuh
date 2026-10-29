#pragma once

#include "ops/softmax_attention/qsa/indexer/work_map.cuh"
#include "ops/softmax_attention/kernels/topk_512.cuh"

namespace ninfer::ops::detail::qsa {

template <int Partition, int Bits = 8, int Copies = 8, int MinBlocks = 1, int MaxQueries = 64,
          bool Reverse = false>
__global__ __launch_bounds__(256, MinBlocks) void indexer_partition_kernel(
    const __grid_constant__ IndexerArguments a) {
    if (a.columns <= kIndexerSelected) return;
    __shared__ IndexerJobs<MaxQueries> jobs;
    __shared__ Top512Storage<Bits, Copies> radix;
    if (!jobs.template build<1, Partition>(a)) return;
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
        select_top512(keys, count, radix, Top512StoreKeys{out});
        __syncthreads();
    }
}

template <int Partition, int Bits = 8, int Copies = 8,
          int MaxPartitions = kIndexerMaxBlocks / Partition>
__global__
__launch_bounds__(256) void indexer_finish_kernel(const __grid_constant__ IndexerArguments a) {
    __shared__ Top512Storage<Bits, Copies> radix;
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
    select_top512(keys, count, radix, Top512StoreIds{out});
}
} // namespace ninfer::ops::detail::qsa
