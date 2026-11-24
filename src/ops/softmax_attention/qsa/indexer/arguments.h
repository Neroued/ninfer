#pragma once

#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail::qsa {

inline constexpr int kIndexerMaxWindowQueries = 128;
inline constexpr int kIndexerSelected         = 512;
inline constexpr int kIndexerKeyTile          = 64;

struct IndexerArguments {
    static constexpr bool ragged = false;
    const __nv_bfloat16 *query, *keys;
    const std::int32_t *positions, *valid, *table_rows;
    const std::int64_t* tables;
    std::int64_t key_row_stride, table_stride;
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

    __device__ IndexerArguments rows(int first, int count) const {
        auto group        = *this;
        const auto column = std::int64_t(first) * tokens;
        const auto slot   = std::int64_t(first) * width;
        group.query += column * 512;
        group.positions += column;
        group.valid += first;
        group.table_rows += first;
        group.scores += slot * columns;
        if (group.candidates) group.candidates += slot * partitions * kIndexerSelected;
        group.ids += column * kIndexerSelected;
        group.counts += column;
        group.batch = count;
        return group;
    }

    __device__ const __nv_bfloat16* key(int b, int g) const {
        const auto offset = tables[g / 16 + std::int64_t(table_rows[b]) * table_stride];
        return reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const std::uint8_t*>(keys) +
                                                      offset) +
               (g % 16) * key_row_stride;
    }
};

struct RaggedIndexerArguments : IndexerArguments {
    const int *token_to_seq, *cu_seqlens;
    int segments;
    static constexpr bool ragged = true;

    __device__ std::int64_t column(int slot) const { return begin + slot; }

    __device__ int visible(int slot) const { return (positions[column(slot)] + 1) / 4; }
};

template <class Arguments>
__device__ int indexer_segment(const Arguments& a, int slot, int row) {
    if constexpr (Arguments::ragged)
        return a.token_to_seq[a.column(slot)];
    else
        return row;
}

} // namespace ninfer::ops::detail::qsa
