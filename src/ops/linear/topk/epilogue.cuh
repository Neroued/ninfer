#pragma once
#include "ops/common/cuda/reduce.cuh"
#include <cub/warp/warp_merge_sort.cuh>

namespace ninfer::ops::detail {
struct TopKHeadOutput {
    std::uint64_t* keys;
    const int* map;
    int groups, begin, domain;

    __device__ std::uint64_t key(float score, int row, int rows) const {
        if (row >= rows || begin + row >= domain) return 0;
        return common::score_id_order_key(score, map ? map[row] : begin + row);
    }

    __device__ void publish(int group, int token, int rank, std::uint64_t value) const {
        keys[(std::int64_t(token) * groups + group) * 16 + rank] = value;
    }
};

template <int Items>
__device__ __forceinline__ std::uint64_t topk_warp_rank(std::uint64_t (&key)[Items]) {
    const int lane       = threadIdx.x & 31;
    std::uint64_t chosen = 0;
#pragma unroll
    for (int rank = 0; rank < 16; ++rank) {
        std::uint64_t best = 0;
#pragma unroll
        for (int i = 0; i < Items; ++i) best = max(best, key[i]);
        const auto high = __reduce_max_sync(0xffffffff, static_cast<unsigned>(best >> 32));
        const auto low  = __reduce_max_sync(0xffffffff, static_cast<unsigned>(best >> 32) == high
                                                            ? static_cast<unsigned>(best)
                                                            : 0u);
        const auto selected = (std::uint64_t(high) << 32) | low;
        if (lane == rank) chosen = selected;
#pragma unroll
        for (int i = 0; i < Items; ++i)
            if (key[i] == selected) key[i] = 0;
    }
    return chosen;
}

template <class S>
using TopKWarpSort = cub::WarpMergeSort<std::uint64_t, S::kBlockRows / 32, 32>;

template <class S, bool RepeatedMax = false, class Out>
__device__ void select_topk_columns(Out out, const float* scores,
                                    typename TopKWarpSort<S>::TempStorage* sort, int row0,
                                    int token0, int rows, int count) {
    constexpr int BR = S::kBlockRows;
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    for (int c = warp; c < count; c += S::kThreads / 32) {
        std::uint64_t key[BR / 32];
#pragma unroll
        for (int i = 0; i < BR / 32; ++i) {
            const int r = lane * (BR / 32) + i;
            key[i]      = out.key(scores[c * BR + r], row0 + r, rows);
        }
        if constexpr (RepeatedMax) {
            const auto chosen = topk_warp_rank(key);
            if (lane < 16) out.publish(row0 / BR, token0 + c, lane, chosen);
        } else {
            TopKWarpSort<S>(sort[warp]).Sort(key, common::ScoreIdOrderGreater{});
#pragma unroll
            for (int i = 0; i < BR / 32; ++i) {
                const int rank = lane * (BR / 32) + i;
                if (rank < 16) out.publish(row0 / BR, token0 + c, rank, key[i]);
            }
        }
    }
}

// Contraction registers survive across column panels. The bounded shared footprint
// permits row reuse without reducing mainloop residency merely to hold all scores.
template <int MaxPanel, bool RepeatedMax = false>
struct TopKPanelEpilogue {
    enum { kTmaCollective = true, kPredicatedRows = true };
    template <class S>
    static constexpr int kPanel = S::kBlockTokens < MaxPanel ? S::kBlockTokens : MaxPanel;
    template <class S>
    static constexpr int kSharedBytes =
        S::kBlockRows * kPanel<S> * sizeof(float) +
        (S::kThreads / 32) * sizeof(typename TopKWarpSort<S>::TempStorage);

    template <class S, bool Full, class Out>
    __device__ void finish_tile(Out out, unsigned char* scratch,
                                float (&acc)[S::kMmaRows][S::kMmaTokens][4], int row0, int token0,
                                int rows, int tokens) const {
        __syncthreads();
        constexpr int BR = S::kBlockRows, BT = S::kBlockTokens, Panel = kPanel<S>;
        constexpr int ColWarps = BT / S::kWarpTokens;
        constexpr int Producer = []() {
            if constexpr (requires { S::kProducerThreads; })
                return S::kProducerThreads;
            else
                return 0;
        }();
        auto* scores = reinterpret_cast<float*>(scratch);
        auto* sort = reinterpret_cast<typename TopKWarpSort<S>::TempStorage*>(scores + BR * Panel);
        const int tid = threadIdx.x, lane = tid & 31;
        const int consumer = tid - Producer, wr = (consumer / 32) / ColWarps,
                  wt = (consumer / 32) % ColWarps;
#pragma unroll 1
        for (int panel = 0; panel < BT; panel += Panel) {
            if (token0 + panel >= tokens) break;
            if (tid >= Producer) {
#pragma unroll
                for (int mi = 0; mi < S::kMmaRows; ++mi) {
                    const int r = wr * S::kWarpRows + mi * 16 + (lane >> 2);
#pragma unroll
                    for (int ni = 0; ni < S::kMmaTokens; ++ni) {
                        const int c = wt * S::kWarpTokens + ni * 8 + 2 * (lane & 3);
                        if (c >= panel && c < panel + Panel) {
                            scores[(c - panel) * BR + r]         = acc[mi][ni][0];
                            scores[(c - panel) * BR + r + 8]     = acc[mi][ni][2];
                            scores[(c + 1 - panel) * BR + r]     = acc[mi][ni][1];
                            scores[(c + 1 - panel) * BR + r + 8] = acc[mi][ni][3];
                        }
                    }
                }
            }
            __syncthreads();
            select_topk_columns<S, RepeatedMax>(
                out, scores, sort, row0, token0 + panel, rows,
                min(min(Panel, BT - panel), tokens - token0 - panel));
            __syncthreads();
        }
    }
};

} // namespace ninfer::ops::detail
