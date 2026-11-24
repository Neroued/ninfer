#pragma once

#include "ops/softmax_attention/qsa/indexer/work_map.cuh"
#include "ops/softmax_attention/kernels/topk_512.cuh"

namespace ninfer::ops::detail::qsa {

template <class Arguments>
__device__ __forceinline__ void indexer_store_all(const Arguments& a, int slot, int visible) {
    const auto column = a.column(slot);
    const int tid     = threadIdx.x;
    if (tid == 0) a.counts[column] = visible;
    if (visible == 0) return;
    auto* out      = a.ids + column * kIndexerSelected;
    int vector_end = 0;
    if (reinterpret_cast<std::uintptr_t>(out) % alignof(int4) == 0) {
        vector_end = (visible / 4) * 4;
        for (int i = 4 * tid; i < vector_end; i += 4 * 256)
            reinterpret_cast<int4*>(out)[i / 4] = make_int4(i, i + 1, i + 2, i + 3);
    }
    for (int i = vector_end + tid; i < visible; i += 256) out[i] = i;
}

__global__
__launch_bounds__(256) void indexer_select_all_kernel(const __grid_constant__ IndexerArguments a) {
    const int slots = a.width * a.batch;
    for (std::int64_t slot = blockIdx.x; slot < slots; slot += gridDim.x)
        indexer_store_all(a, slot, a.visible(slot));
}

__global__ __launch_bounds__(256) void indexer_select_all_ragged_kernel(
    const __grid_constant__ RaggedIndexerArguments a) {
    for (std::int64_t slot = blockIdx.x; slot < a.width; slot += gridDim.x)
        indexer_store_all(a, slot, a.visible(slot));
}

inline void launch_indexer_select_all(RaggedIndexerArguments a, int workers, cudaStream_t stream) {
    a.width = a.tokens;
    indexer_select_all_ragged_kernel<<<workers, 256, 0, stream>>>(a);
}

__global__ void indexer_idle_kernel() {}

inline void launch_indexer_select_all(IndexerArguments a, int workers, cudaStream_t stream) {
    indexer_select_all_kernel<<<workers, 256, 0, stream>>>(a);
    indexer_idle_kernel<<<1, 1, 0, stream>>>();
    indexer_idle_kernel<<<1, 1, 0, stream>>>();
}

template <int Partition>
__device__ __forceinline__ void indexer_load_score_keys(std::uint64_t (&keys)[Partition / 256],
                                                        const float* scores, int first, int count) {
#pragma unroll
    for (int j = 0; j < Partition / 256; ++j) {
        const int i = j * 256 + threadIdx.x;
        keys[j]     = i < count ? common::score_id_order_key(scores[i], first + i) : 0;
    }
}

template <int Partition, int Bits, int Copies, class Arguments = IndexerArguments>
__global__
__launch_bounds__(256) void indexer_single_select_kernel(const __grid_constant__ Arguments a) {
    __shared__ Top512Storage<Bits, Copies> radix;
    const int slots = a.width * a.batch;
    for (std::int64_t slot = blockIdx.x; slot < slots; slot += gridDim.x) {
        const int visible = a.visible(slot);
        if (visible <= kIndexerSelected) {
            indexer_store_all(a, slot, visible);
            continue;
        }
        std::uint64_t keys[Partition / 256];
        indexer_load_score_keys<Partition>(keys, a.scores + slot * a.columns, 0, visible);
        const auto column = a.column(slot);
        select_top512(keys, visible, radix, Top512StoreIds{a.ids + column * kIndexerSelected});
        if (threadIdx.x == 0) a.counts[column] = kIndexerSelected;
        __syncthreads();
    }
}

template <int Partition, int Bits, int Copies, int MinBlocks, int MaxQueries, bool Reverse = false,
          bool BatchedLoop = false, class Arguments = IndexerArguments>
__global__
__launch_bounds__(256,
                  MinBlocks) void indexer_partition_kernel(const __grid_constant__ Arguments a) {
    __shared__ std::conditional_t<Arguments::ragged, RaggedIndexerSlots<MaxQueries>,
                                  IndexerJobs<MaxQueries>>
        jobs;
    __shared__ Top512Storage<Bits, Copies> radix;
    for_each_indexer_job_group<MaxQueries, BatchedLoop>(
        a, [&](const auto& group, int worker, int stride) {
        if (!jobs.template build<1, Partition>(group)) return;
        const int total = jobs.prefix[jobs.groups];
        for (int step = worker; step < total; step += stride) {
            const int job  = Reverse ? total - 1 - step : step;
            const int slot = jobs.group_for(job), part = job - jobs.prefix[slot];
            const int first   = part * Partition;
            const int visible = jobs.visible[slot];
            const int count   = min(Partition, visible - first);
            std::uint64_t keys[Partition / 256];
            indexer_load_score_keys<Partition>(
                keys, group.scores + std::int64_t(slot) * group.columns + first, first, count);
            if (visible <= Partition) {
                const auto column = group.column(slot);
                select_top512(keys, count, radix,
                              Top512StoreIds{group.ids + column * kIndexerSelected});
                if (threadIdx.x == 0) group.counts[column] = kIndexerSelected;
            } else {
                // A discarded key has 512 better keys in its own partition, so it
                // cannot belong to the global Top512.
                auto* out = group.candidates +
                            (std::int64_t(slot) * group.partitions + part) * kIndexerSelected;
                select_top512(keys, count, radix, Top512StoreKeys{out});
            }
            __syncthreads();
        }
    });
}

template <int Partition, int Bits, int Copies, int MaxPartitions,
          class Arguments = IndexerArguments>
__global__ __launch_bounds__(256) void indexer_finish_kernel(const __grid_constant__ Arguments a) {
    __shared__ Top512Storage<Bits, Copies> radix;
    const int slot = blockIdx.x, tid = threadIdx.x, visible = a.visible(slot);
    if (visible <= kIndexerSelected) {
        indexer_store_all(a, slot, visible);
        return;
    }
    if (visible <= Partition) return; // The partition kernel stored this row's final selection.

    const int parts = (visible + Partition - 1) / Partition;
    const int count = parts * kIndexerSelected;
    // Mask the unwritten suffix of the last partition before reading reused scratch.
    const int valid =
        count - kIndexerSelected + min(kIndexerSelected, visible - (parts - 1) * Partition);
    const auto* input   = a.candidates + std::int64_t(slot) * a.partitions * kIndexerSelected;
    constexpr int Items = MaxPartitions * kIndexerSelected / 256;
    std::uint64_t keys[Items];
#pragma unroll
    for (int j = 0; j < Items; ++j) {
        const int i = j * 256 + tid;
        keys[j]     = i < valid ? input[i] : 0;
    }
    const auto column = a.column(slot);
    if (tid == 0) a.counts[column] = kIndexerSelected;
    select_top512(keys, count, radix, Top512StoreIds{a.ids + column * kIndexerSelected});
}

template <int Bits, int Copies, class Arguments>
void launch_indexer_single_select(Arguments a, int workers, cudaStream_t stream) {
    indexer_single_select_kernel<kIndexerPartition, Bits, Copies><<<workers, 256, 0, stream>>>(a);
    if constexpr (!Arguments::ragged) indexer_idle_kernel<<<1, 1, 0, stream>>>();
}

template <int Bits, int Copies, class Arguments>
void launch_indexer_finish(Arguments a, cudaStream_t stream) {
    const int slots = a.width * a.batch;
    if (a.partitions <= 4)
        indexer_finish_kernel<kIndexerPartition, Bits, Copies, 4><<<slots, 256, 0, stream>>>(a);
    else
        indexer_finish_kernel<kIndexerPartition, Bits, Copies, 16><<<slots, 256, 0, stream>>>(a);
}
} // namespace ninfer::ops::detail::qsa
