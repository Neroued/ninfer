#pragma once

#include "ops/softmax_attention/kpool_mla/indexer/arguments.h"
#include "ops/softmax_attention/kernels/topk_512.cuh"
#include "ops/softmax_attention/kernels/topk_512_words.cuh"
#include "ops/softmax_attention/kpool_mla/indexer/work.cuh"

#include <type_traits>

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

// FP8 score windows can contain very different live histories inside one captured
// envelope. Enumerate actual partitions so short requests share all resident CTAs.
template <int MinBlocks>
__global__ __launch_bounds__(256, MinBlocks) void indexer_partition_fp8_kernel(
    const __grid_constant__ IndexerArguments a) {
    if (a.columns <= kIndexSelected) return;
    __shared__ Top512WordStorage radix;
    const int tid = threadIdx.x;
    int jobs      = 0;
    bool dense    = true;
    for (int b = 0; b < a.batch; ++b) {
        const auto span = partition_span(a, b);
        jobs += span.jobs;
        dense &= span.first == 0 && span.live == a.width && span.low_rows == a.width &&
                 span.parts == a.partitions;
    }
    for (int job = blockIdx.x; job < jobs; job += gridDim.x) {
        const auto mapping =
            dense ? PartitionJob{job / a.partitions, job % a.partitions} : partition_job(a, job);
        const int slot    = __reduce_or_sync(0xffffffffu, unsigned(mapping.slot));
        const int part    = __reduce_or_sync(0xffffffffu, unsigned(mapping.part));
        const auto column = a.column(slot);
        const int visible = a.ends[column] / 4, first = part * kIndexPartition;
        const int count = min(kIndexPartition, visible - first);
        unsigned scores[kIndexPartition / 256];
        const Top512DenseIds ids{first};
#pragma unroll
        for (int j = 0; j < kIndexPartition / 256; ++j) {
            const int i = j * 256 + tid;
            scores[j]   = i < count ? common::ordered_score_bits(
                                        a.scores[std::int64_t(slot) * a.columns + first + i])
                                    : 0;
        }
        if (visible <= kIndexPartition)
            select_top512_words(scores, ids, count, radix,
                                Top512StoreIds{a.ids + column * kIndexSelected});
        else
            select_top512_words(
                scores, ids, count, radix,
                Top512StoreKeys{a.candidates +
                                (std::int64_t(slot) * a.partitions + part) * kIndexSelected});
        __syncthreads();
    }
}

template <bool Final, int Bits = 6, int Copies = 2, bool WordKeys = false>
__global__
__launch_bounds__(256) void indexer_merge_kernel(const __grid_constant__ IndexerArguments a) {
    if (a.partitions <= (Final ? 1 : kIndexFanIn)) return;
    __shared__ std::conditional_t<WordKeys, Top512WordStorage, Top512Storage<Bits, Copies>> radix;
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
    constexpr int Items = kIndexFanIn * kIndexSelected / 256;
    const auto store    = [&] {
        if constexpr (Final)
            return Top512StoreIds{a.ids + column * kIndexSelected};
        else
            return Top512StoreKeys{a.merged +
                                   (std::int64_t(slot) * a.merge_groups + group) * kIndexSelected};
    }();
    if constexpr (WordKeys) {
        unsigned scores[Items];
        Top512GatherIds<Items> ids;
#pragma unroll
        for (int j = 0; j < Items; ++j) {
            const int i    = j * 256 + tid;
            const auto key = i < count ? input[i] : 0;
            scores[j]      = unsigned(key >> 32);
            ids.values[j]  = unsigned(key);
        }
        select_top512_words(scores, ids, count, radix, store);
    } else {
        std::uint64_t keys[Items];
#pragma unroll
        for (int j = 0; j < Items; ++j) {
            const int i = j * 256 + tid;
            keys[j]     = i < count ? input[i] : 0;
        }
        select_top512(keys, count, radix, store);
    }
}
} // namespace ninfer::ops::detail::kpool
