#pragma once

#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail::kpool {

inline constexpr int kIndexPartition = 4096;
inline constexpr int kIndexSelected  = 512;
inline constexpr int kIndexFanIn     = 8;

struct IndexerArguments {
    const __nv_bfloat16 *query, *keys;
    const float* coefficients;
    const std::int32_t *positions, *valid, *table_rows;
    const std::int64_t* tables;
    std::int64_t table_stride;
    float* scores;
    std::uint64_t *candidates, *merged;
    std::int32_t *ids, *counts, *ends;
    int tokens, batch, begin, width, columns, partitions, merge_groups;

    __device__ std::int64_t column(int slot) const {
        return begin + slot % width + std::int64_t(slot / width) * tokens;
    }

    __device__ int end(int slot) const {
        const int b = slot / width, t = begin + slot % width;
        return t < valid[b] ? positions[t + std::int64_t(b) * tokens] + 1 : 0;
    }
};

inline constexpr int kIndexMaxWindowQueries = 256;

struct PackedIndexerArguments : IndexerArguments {
    const int* token_to_seq;
    // Each entry stores a local start and a length, independent of replay metadata.
    std::uint16_t tiles[kIndexMaxWindowQueries];
    int tile_count;

    __device__ std::int64_t column(int slot) const { return std::int64_t(begin) + slot; }

    __device__ int end(int slot) const { return positions[column(slot)] + 1; }
};

struct Fp8IndexerArguments {
    IndexerArguments index;
    const std::uint8_t *keys, *scales;
    std::uint8_t* query;
    float* coefficients;
    int query_begin, query_width;
};

struct PackedFp8IndexerArguments {
    PackedIndexerArguments index;
    const std::uint8_t *keys, *scales;
    std::uint8_t* query;
    float* coefficients;
};
} // namespace ninfer::ops::detail::kpool
