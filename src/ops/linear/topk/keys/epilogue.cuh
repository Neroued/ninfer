#pragma once
#include "ops/linear/topk/keys/dispatch.h"
#include "ops/linear/topk/epilogue.cuh"

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

struct SlicedEpilogue {
    enum { kSlicedTile = true };
    template <class S>
    static constexpr int kSharedBytes =
        S::kBlockRows * S::kBlockTokens * sizeof(float) +
        (S::kThreads / 32) * sizeof(typename TopKWarpSort<S>::TempStorage);

    template <class S>
    __device__ void finish_sliced_tile(Output out, float* scores, int row0, int token0, int rows,
                                       int tokens) const {
        static_assert(kSharedBytes<S> <= 99 * 1024);
        auto* sort = reinterpret_cast<typename TopKWarpSort<S>::TempStorage*>(
            scores + S::kBlockRows * S::kBlockTokens);
        select_topk_columns<S>(out, scores, sort, row0, token0, rows,
                               min(S::kBlockTokens, tokens - token0));
    }
};
} // namespace ninfer::ops::detail::topk_keys
