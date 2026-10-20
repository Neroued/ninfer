#pragma once

#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail::qsa {

inline constexpr int kIndexerMaxVisibleTokens = 262144;
inline constexpr int kIndexerMaxBlocks        = kIndexerMaxVisibleTokens / 4;
inline constexpr int kIndexerMaxWindowQueries = 128;
inline constexpr int kIndexerSelected         = 512;
inline constexpr int kIndexerKeyTile          = 64;

struct IndexerArguments {
    const __nv_bfloat16 *query, *keys;
    const std::int32_t *positions, *valid, *table_rows, *tables;
    std::int64_t key_row_stride, key_page_stride, table_stride;
    float* scores;
    std::uint64_t* candidates;
    std::int32_t *ids, *counts;
    int tokens, batch, begin, width, columns, partitions;

    __device__ std::int64_t column(int slot) const {
        return begin + slot % width + std::int64_t(slot / width) * tokens;
    }

    __device__ int visible(int slot) const {
        const int t = begin + slot % width, b = slot / width;
        return t < valid[b] ? (positions[t + std::int64_t(b) * tokens] + 1) / 4 : 0;
    }

    __device__ const __nv_bfloat16* key(int b, int g) const {
        const int page = tables[g / 16 + std::int64_t(table_rows[b]) * table_stride];
        return keys + (g % 16) * key_row_stride + std::int64_t(page) * key_page_stride;
    }
};

} // namespace ninfer::ops::detail::qsa
