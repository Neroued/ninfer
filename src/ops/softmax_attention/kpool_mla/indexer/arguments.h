#pragma once

#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail::kpool {

inline constexpr int kIndexPartition = 4096;
inline constexpr int kIndexSelected  = 512;
inline constexpr int kIndexFanIn     = 8;
inline constexpr int kIndexGraphRows = 16;

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

struct Fp8IndexerArguments {
    IndexerArguments index;
    const std::uint8_t *keys, *scales;
    std::uint8_t* query;
    float* coefficients;
    int query_begin, query_width;
};
} // namespace ninfer::ops::detail::kpool
