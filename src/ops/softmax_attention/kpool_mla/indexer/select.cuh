#pragma once

#include "ops/softmax_attention/kpool_mla/indexer/arguments.h"
#include "ops/softmax_attention/kernels/topk_512.cuh"

namespace ninfer::ops::detail::kpool {

__device__ inline int candidate_count(int count, int partition) {
    return (count - 1) / partition * kIndexSelected +
           min(kIndexSelected, (count - 1) % partition + 1);
}

template <int Bits = 6, int Copies = 2>
__global__
__launch_bounds__(256) void indexer_partition_kernel(const __grid_constant__ IndexerArguments a) {
    if (a.columns <= kIndexSelected) return;
    __shared__ Top512Storage<Bits, Copies> radix;
    const int tid = threadIdx.x;
    for (int job = blockIdx.x; job < a.width * a.batch * a.partitions; job += gridDim.x) {
        const int slot = job / a.partitions, part = job % a.partitions;
        const auto column = a.column(slot);
        const int visible = a.ends[column] / 4, first = part * kIndexPartition;
        if (visible <= kIndexSelected || first >= visible) continue;
        const int count = min(kIndexPartition, visible - first);
        std::uint64_t keys[kIndexPartition / 256];
#pragma unroll
        for (int j = 0; j < kIndexPartition / 256; ++j) {
            const int i = j * 256 + tid;
            keys[j]     = i < count
                              ? common::score_id_order_key(
                                a.scores[std::int64_t(slot) * a.columns + first + i], first + i)
                              : 0;
        }
        if (visible <= kIndexPartition)
            select_top512(keys, count, radix, Top512StoreIds{a.ids + column * kIndexSelected});
        else
            select_top512(
                keys, count, radix,
                Top512StoreKeys{a.candidates +
                                (std::int64_t(slot) * a.partitions + part) * kIndexSelected});
        __syncthreads();
    }
}

template <bool Final, int Bits = 6, int Copies = 2>
__global__
__launch_bounds__(256) void indexer_merge_kernel(const __grid_constant__ IndexerArguments a) {
    if (a.partitions <= (Final ? 1 : kIndexFanIn)) return;
    __shared__ Top512Storage<Bits, Copies> radix;
    const int tid     = threadIdx.x;
    const int slot    = Final ? blockIdx.x : blockIdx.x / a.merge_groups;
    const int group   = Final ? 0 : blockIdx.x % a.merge_groups;
    const auto column = a.column(slot);
    const int visible = a.ends[column] / 4;
    if (visible <= kIndexPartition) return;
    const int parts = (visible + kIndexPartition - 1) / kIndexPartition;
    if (!Final && (parts <= kIndexFanIn || group * kIndexFanIn >= parts)) return;
    const std::uint64_t* input;
    int count;
    if constexpr (Final) {
        if (parts <= kIndexFanIn) {
            input = a.candidates + std::int64_t(slot) * a.partitions * kIndexSelected;
            count = candidate_count(visible, kIndexPartition);
        } else {
            input = a.merged + std::int64_t(slot) * a.merge_groups * kIndexSelected;
            count = candidate_count(visible, kIndexPartition * kIndexFanIn);
        }
    } else {
        input = a.candidates +
                (std::int64_t(slot) * a.partitions + group * kIndexFanIn) * kIndexSelected;
        const int remaining =
            min(visible - group * kIndexFanIn * kIndexPartition, kIndexFanIn * kIndexPartition);
        count = candidate_count(remaining, kIndexPartition);
    }
    std::uint64_t keys[kIndexFanIn * kIndexSelected / 256];
#pragma unroll
    for (int j = 0; j < kIndexFanIn * kIndexSelected / 256; ++j) {
        const int i = j * 256 + tid;
        keys[j]     = i < count ? input[i] : 0;
    }
    if constexpr (Final)
        select_top512(keys, count, radix, Top512StoreIds{a.ids + column * kIndexSelected});
    else
        select_top512(keys, count, radix,
                      Top512StoreKeys{a.merged + (std::int64_t(slot) * a.merge_groups + group) *
                                                     kIndexSelected});
}
} // namespace ninfer::ops::detail::kpool
