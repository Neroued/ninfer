#pragma once

#include "ops/sampling/kernels/distribution.cuh"
#include <bit>
#include "ops/common/cuda/pipeline.cuh"

namespace ninfer::ops::sampling_detail {

template <int Threads, int Tile>
struct alignas(16) SamplingInputTile {
    static constexpr int kItems = Tile / Threads;
    static_assert(kItems % 8 == 0);
    __nv_bfloat16 logits[Tile];
    int counts[Tile];

    __device__ void prefetch(const __nv_bfloat16* input, const int* history, int available,
                             bool penalties) {
        const int tid = threadIdx.x;
        if ((reinterpret_cast<std::uintptr_t>(input) & 15U) == 0) {
#pragma unroll
            for (int i = 0; i < kItems / 8; ++i) {
                const int first = (i * Threads + tid) * 8;
                cp_async_zfill<16, Cache::cg>(logits + first, input + first,
                                              min(16, max(0, (available - first) * 2)));
            }
            cp_commit();
        } else {
#pragma unroll
            for (int i = 0; i < kItems; ++i) {
                const int at = i * Threads + tid;
                logits[at]   = at < available ? input[at] : __float2bfloat16(0.F);
            }
        }
        if (penalties && history) {
            if ((reinterpret_cast<std::uintptr_t>(history) & 15U) == 0) {
#pragma unroll
                for (int i = 0; i < kItems / 4; ++i) {
                    const int first = (i * Threads + tid) * 4;
                    cp_async_zfill<16, Cache::cg>(counts + first, history + first,
                                                  min(16, max(0, (available - first) * 4)));
                }
                cp_commit();
            } else {
#pragma unroll
                for (int i = 0; i < kItems; ++i) {
                    const int at = i * Threads + tid;
                    cp_async_zfill<4, Cache::ca>(counts + at, history + at, at < available ? 4 : 0);
                    if (i % 4 == 3) cp_commit();
                }
            }
        } else if (penalties) {
#pragma unroll
            for (int i = 0; i < kItems; ++i) counts[i * Threads + tid] = 0;
        }
    }
};

template <int Tile>
struct DraftCountTile {
    static constexpr int kMaxPrefix = 255;
    unsigned int words[Tile / 4];

    __device__ void prepare(TokenRow drafts, int prefix, int begin) {
        for (int i = threadIdx.x; i < Tile / 4; i += blockDim.x) words[i] = 0;
        __syncthreads();
        for (int j = threadIdx.x; j < prefix; j += blockDim.x) {
            const int local = drafts[j] - begin;
            if (local >= 0 && local < Tile) atomicAdd(&words[local / 4], 1U << (8 * (local & 3)));
        }
        __syncthreads();
    }

    __device__ int count(int local) const { return (words[local / 4] >> (8 * (local & 3))) & 255U; }
};

template <int Threads, int Items, class KeyType = unsigned long long>
struct ThresholdSelection {
    static constexpr int kThreads = Threads, kItems = Items, kTile = Threads * Items;
    static_assert(Threads > kSamplerCandidateCap && std::has_single_bit(unsigned(Threads)));
    static_assert(std::has_single_bit(unsigned(kTile)) && kTile <= 65536);
    using Key                                        = KeyType;
    static constexpr int kSamplingTileIndexBits      = std::countr_zero(unsigned(kTile));
    static constexpr unsigned kSamplingTileIndexMask = kTile - 1;
    using MaxSort                                    = cub::BlockMergeSort<Key, Threads, 1>;
    using SmallSort                                  = cub::WarpMergeSort<Key, 1>;
    using FinalSort                                  = cub::WarpMergeSort<Key, 4>;

    struct Storage {
        union {
            typename MaxSort::TempStorage maxima;
            typename SmallSort::TempStorage small;
            typename FinalSort::TempStorage final;
            Key warp_maxima[Threads / 32];
        } sort;

        Key candidates[kSamplerCandidateCap * Items];
        Key scratch[kSamplerCandidateCap *
                    ((kSamplerCandidateCap * Items + Threads - 1) / Threads)];
        Key threshold;
        int count;
    };

    __device__ static Key threshold(Key maximum, int cap, Storage& storage) {
        Key values[1]{maximum};
        MaxSort(storage.sort.maxima).Sort(values, ScoreIdOrderGreater{});
        if (threadIdx.x == cap - 1) storage.threshold = values[0];
        __syncthreads();
        return storage.threshold;
    }

    __device__ static void append(Key key, Key lower, Key* output, Storage& storage) {
        const int lane       = threadIdx.x & 31;
        const unsigned valid = __ballot_sync(0xffffffffU, key && key >= lower);
        if (valid) {
            const int leader = __ffs(valid) - 1;
            int first        = 0;
            if (lane == leader) first = atomicAdd(&storage.count, __popc(valid));
            first = __shfl_sync(0xffffffffU, first, leader);
            if (valid & (1U << lane)) output[first + __popc(valid & ((1U << lane) - 1))] = key;
        }
    }

    __device__ __forceinline__ static void select_values(Key (&values)[Items], int cap, Key* output,
                                                         Storage& storage) {
        const int tid = threadIdx.x;
        Key maximum   = 0;
#pragma unroll
        for (int i = 0; i < Items; ++i) maximum = max(maximum, values[i]);
        if (cap == 1) {
            if constexpr (sizeof(Key) == 8)
                maximum = block_reduce_max_key(maximum, storage.sort.warp_maxima);
            else {
                maximum = __reduce_max_sync(0xffffffffU, maximum);
                if ((tid & 31) == 0) storage.sort.warp_maxima[tid / 32] = maximum;
                __syncthreads();
                maximum = tid < Threads / 32 ? storage.sort.warp_maxima[tid] : 0;
                if (tid < 32) maximum = __reduce_max_sync(0xffffffffU, maximum);
            }
            if (tid == 0) output[0] = maximum;
            return;
        }

        // The cap-th thread maximum is a lower bound on the exact top-cap.
        // Nonzero survivors come from at most cap thread lists of Items keys.
        Key lower = threshold(maximum, cap, storage);
        if (tid == 0) storage.count = 0;
        __syncthreads();
#pragma unroll
        for (int i = 0; i < Items; ++i) append(values[i], lower, storage.candidates, storage);
        __syncthreads();

        int current = 0;
        while (storage.count > 128) {
            const int count = storage.count;
            maximum         = 0;
            for (int i = tid; i < count; i += Threads)
                maximum = max(maximum, (current ? storage.scratch : storage.candidates)[i]);
            lower = threshold(maximum, cap, storage);
            if (tid == 0) storage.count = 0;
            __syncthreads();
            for (int first = 0; first < count; first += Threads) {
                const int index = first + tid;
                const Key key =
                    index < count ? (current ? storage.scratch : storage.candidates)[index] : 0;
                append(key, lower, (current ? storage.candidates : storage.scratch), storage);
            }
            __syncthreads();
            current = 1 - current;
        }

        if (tid >= 32) return;
        const int count = storage.count;
        if (count <= 32) {
            Key keys[1]{tid < count ? (current ? storage.scratch : storage.candidates)[tid] : 0};
            SmallSort(storage.sort.small).Sort(keys, ScoreIdOrderGreater{});
            if (tid < cap) output[tid] = keys[0];
        } else {
            Key keys[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int index = tid * 4 + i;
                keys[i] =
                    index < count ? (current ? storage.scratch : storage.candidates)[index] : 0;
            }
            FinalSort(storage.sort.final).Sort(keys, ScoreIdOrderGreater{});
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int index = tid * 4 + i;
                if (index < cap) output[index] = keys[i];
            }
        }
    }
};

} // namespace ninfer::ops::sampling_detail
