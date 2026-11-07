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
    const __nv_bfloat16 *query, *latent;
    const void *key_up, *value_up;
    const std::uint8_t *key_scales, *value_scales;
    bool key_mxfp8 = false, value_mxfp8 = false;
    const std::uint8_t *latent_fp8, *latent_scales;
    std::uint8_t *query_fp8, *query_scales;
    const std::int32_t *valid, *table_rows, *selection_rows, *ids, *counts, *ends;
    const std::int64_t* tables;
    std::int64_t table_stride, begin;
    __nv_bfloat16 *mapped, *out;
    float *numerator, *maximum, *denominator;
    int tokens, batch, width, splits, tile;
    int preferred_splits = 0, split_wave_size = 0;
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

    // Metadata is stable throughout the call. All stages derive the same compact
    // partial layout without a host readback or an additional device pass.
    __device__ int execution_splits() const {
        if (!preferred_splits) return splits;
        int live = 0;
        for (int b = 0; b < batch; ++b) {
            const auto row_begin = std::int64_t(b) * tokens;
            const auto row_end   = row_begin + valid[b];
            const auto first     = begin > row_begin ? begin : row_begin;
            const auto last      = begin + width < row_end ? begin + width : row_end;
            if (last > first) live += int(last - first);
        }
        return live * splits <= split_wave_size ? splits : preferred_splits;
    }

    __device__ std::int64_t partial(int local, int split, int head, int stride) const {
        return (std::int64_t(local) * stride + split) * 64 + head;
    }
};
} // namespace ninfer::ops::detail::kpool
