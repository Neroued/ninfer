#pragma once

#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::kpool {

inline constexpr int kMlaWindow = 1024;

struct MlaWork {
    int row, end, pools, size;
};

struct SparseMlaArguments {
    const __nv_bfloat16 *query, *key_up, *value_up, *latent;
    const std::int32_t *valid, *table_rows, *selection_rows, *ids, *counts, *ends;
    const std::int64_t* tables;
    std::int64_t table_stride, begin;
    __nv_bfloat16 *mapped, *out;
    float *numerator, *maximum, *denominator;
    int tokens, batch, width, splits, tile;
    float scale;
    bool direct              = false;
    std::size_t latent_bytes = 0;

    __device__ MlaWork work(int local) const {
        if (local >= width) return {};
        const auto column = begin + local;
        if (column % tokens >= valid[column / tokens]) return {};
        const int row = selection_rows[column], end = ends[row], pools = counts[row];
        return {row, end, pools, 4 * pools + end % 4};
    }

    __device__ std::int64_t partial(int local, int split, int head) const {
        return (std::int64_t(local) * splits + split) * 64 + head;
    }
};
} // namespace ninfer::ops::detail::kpool
