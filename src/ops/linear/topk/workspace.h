#pragma once

#include "core/layout.h"
#include "ops/common/numeric/math.h"

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kLinearTopK                = 16;
inline constexpr std::int32_t kLinearTopKMaxChunkColumns = 128;
inline constexpr std::int32_t kLinearTopKGroupedRows     = 128;
inline constexpr std::int32_t kLinearTopKDirectRows      = 16;
inline constexpr std::int32_t kLinearTopKMergeFanIn      = 32;
inline constexpr std::int32_t kLinearTopKFullRows        = 248320;
inline constexpr std::int32_t kLinearTopKFullValidRows   = 248077;
inline constexpr std::int32_t kLinearTopKMappedRows      = 131072;
inline constexpr std::int32_t kLinearTopKHidden          = 5120;

inline bool linear_topk_chunk_reachable(int columns, int first, int last) {
    bool reachable = columns >= first && columns <= last;
    if (last > kLinearTopKMaxChunkColumns) {
        if (columns == kLinearTopKMaxChunkColumns) return true;
        const auto next = std::int64_t(first) + (columns - first % kLinearTopKMaxChunkColumns +
                                                 kLinearTopKMaxChunkColumns) %
                                                    kLinearTopKMaxChunkColumns;
        reachable |= next <= last;
    }
    return reachable;
}

struct LinearTopKWorkspace {
    Tensor partial_keys;
    Tensor group_keys;
    Tensor secondary_keys;
    Tensor group_done;
    std::int32_t producer_groups  = 0;
    std::int32_t merge_groups     = 0;
    std::int32_t secondary_groups = 0;
    std::int32_t columns          = 0;
};

template <class Allocator>
LinearTopKWorkspace allocate_linear_topk_workspace(Allocator& allocator, std::int32_t head_rows,
                                                   std::int32_t columns,
                                                   std::int32_t rows_per_producer) {
    const std::int32_t producer_groups  = common::div_up(head_rows, rows_per_producer);
    const std::int32_t merge_groups     = common::div_up(producer_groups, kLinearTopKMergeFanIn);
    const std::int32_t secondary_groups = merge_groups > kLinearTopKMergeFanIn
                                              ? common::div_up(merge_groups, kLinearTopKMergeFanIn)
                                              : 0;
    LinearTopKWorkspace out;
    out.partial_keys = allocator.alloc(DType::I64, {kLinearTopK, producer_groups, columns}, 256);
    out.group_keys   = allocator.alloc(DType::I64, {kLinearTopK, merge_groups, columns}, 256);
    if (secondary_groups != 0) {
        out.secondary_keys =
            allocator.alloc(DType::I64, {kLinearTopK, secondary_groups, columns}, 256);
    }
    out.group_done       = allocator.alloc(DType::I32, {columns}, 256);
    out.producer_groups  = producer_groups;
    out.merge_groups     = merge_groups;
    out.secondary_groups = secondary_groups;
    out.columns          = columns;
    return out;
}

} // namespace ninfer::ops::detail
