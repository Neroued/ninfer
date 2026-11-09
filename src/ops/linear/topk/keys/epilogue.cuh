#pragma once
#include "ops/linear/topk/keys/dispatch.h"
#include "ops/common/cuda/reduce.cuh"
#include <cub/warp/warp_merge_sort.cuh>

namespace ninfer::ops::detail::topk_keys {
struct Output {
    std::uint64_t* keys;
    int groups, begin, domain;

    __device__ std::uint64_t key(float score, int row, int physical_rows) const {
        if (row >= physical_rows) return 0;
        const auto id = begin + row;
        return id < domain ? common::score_id_order_key(score, id) : 0;
    }

    __device__ void publish(int group, int token, int rank, std::uint64_t value) const {
        keys[(std::int64_t(token) * groups + group) * 16 + rank] = value;
    }

    // Sixteen-row producers publish all FP32 keys; the following merge selects.
    __device__ void store(int row, int token, float score) const {
        publish(row / 16, token, row % 16, key(score, row, INT_MAX - begin));
    }
};

inline Output output(const LinearTopKWorkspace& w, int begin, int domain) {
    return {static_cast<std::uint64_t*>(w.partial_keys.data), w.producer_groups, begin, domain};
}
template <class S>
using WarpSort = cub::WarpMergeSort<std::uint64_t, S::kBlockRows / 32, 32>;

template <class S>
__device__ void select_columns(Output out, const float* scores,
                               typename WarpSort<S>::TempStorage* sort, int row0, int token0,
                               int rows, int count) {
    constexpr int BR = S::kBlockRows;
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    for (int c = warp; c < count; c += S::kThreads / 32) {
        std::uint64_t key[BR / 32];
#pragma unroll
        for (int i = 0; i < BR / 32; ++i) {
            const int r = lane * (BR / 32) + i;
            key[i]      = out.key(scores[c * BR + r], row0 + r, rows);
        }
        WarpSort<S>(sort[warp]).Sort(key, common::ScoreIdOrderGreater{});
#pragma unroll
        for (int i = 0; i < BR / 32; ++i) {
            const int rank = lane * (BR / 32) + i;
            if (rank < 16) out.publish(row0 / BR, token0 + c, rank, key[i]);
        }
    }
}

// Contraction registers survive across column panels. The bounded shared footprint
// permits row reuse without reducing mainloop residency merely to hold all scores.
template <int MaxPanel>
struct PanelEpilogue {
    enum { kTmaCollective = true, kPredicatedRows = true };
    template <class S>
    static constexpr int kPanel = S::kBlockTokens < MaxPanel ? S::kBlockTokens : MaxPanel;
    template <class S>
    static constexpr int kSharedBytes =
        S::kBlockRows * kPanel<S> * sizeof(float) +
        (S::kThreads / 32) * sizeof(typename WarpSort<S>::TempStorage);

    template <class S, bool Full>
    __device__ void finish_tile(Output out, unsigned char* scratch,
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
        auto* scores  = reinterpret_cast<float*>(scratch);
        auto* sort    = reinterpret_cast<typename WarpSort<S>::TempStorage*>(scores + BR * Panel);
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
            select_columns<S>(out, scores, sort, row0, token0 + panel, rows,
                              min(min(Panel, BT - panel), tokens - token0 - panel));
            __syncthreads();
        }
    }
};

struct SlicedEpilogue {
    enum { kSlicedTile = true };
    template <class S>
    static constexpr int kSharedBytes =
        S::kBlockRows * S::kBlockTokens * sizeof(float) +
        (S::kThreads / 32) * sizeof(typename WarpSort<S>::TempStorage);

    template <class S>
    __device__ void finish_sliced_tile(Output out, float* scores, int row0, int token0, int rows,
                                       int tokens) const {
        static_assert(kSharedBytes<S> <= 99 * 1024);
        auto* sort = reinterpret_cast<typename WarpSort<S>::TempStorage*>(
            scores + S::kBlockRows * S::kBlockTokens);
        select_columns<S>(out, scores, sort, row0, token0, rows,
                          min(S::kBlockTokens, tokens - token0));
    }
};
} // namespace ninfer::ops::detail::topk_keys
