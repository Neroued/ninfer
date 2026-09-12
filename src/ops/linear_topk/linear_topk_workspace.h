#pragma once

#include "core/layout.h"
#include "ops/common/math.h"

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kLinearTopK                = 16;
inline constexpr std::int32_t kLinearTopKMaxChunkColumns = 128;
inline constexpr std::int32_t kLinearTopKGroupedRows     = 128;
inline constexpr std::int32_t kLinearTopKDirectRows      = 16;
inline constexpr std::int32_t kLinearTopKMergeFanIn      = 32;
inline constexpr std::int32_t kLinearTopKFullRows        = 248320;
inline constexpr std::int32_t kLinearTopKFullValidRows   = 248077;
inline constexpr std::int32_t kLinearTopKOptimizedRows   = 131072;
inline constexpr std::int32_t kLinearTopKHidden          = 5120;

struct LinearTopKWorkspace {
    std::int32_t tile_columns = 0;
    std::int32_t block_k      = 128;
    Tensor partial_keys;
    Tensor group_keys;
    Tensor secondary_keys;
    Tensor group_done;
    std::int32_t producer_groups   = 0;
    std::int32_t merge_groups      = 0;
    std::int32_t secondary_groups  = 0;
    std::int32_t columns           = 0;
    std::int32_t rows_per_producer = 0;
};

template <class Allocator>
LinearTopKWorkspace
allocate_linear_topk_workspace(Allocator& allocator, std::int32_t head_rows, std::int32_t columns,
                               std::int32_t rows_per_producer, std::int32_t tile_columns = 0,
                               std::int32_t block_k = 128) {
    const std::int32_t producer_groups = div_up(head_rows, rows_per_producer);
    const std::int32_t merge_groups    = div_up(producer_groups, kLinearTopKMergeFanIn);
    const std::int32_t secondary_groups =
        merge_groups > kLinearTopKMergeFanIn ? div_up(merge_groups, kLinearTopKMergeFanIn) : 0;
    LinearTopKWorkspace out;
    out.tile_columns = tile_columns;
    out.block_k      = block_k;
    out.partial_keys = allocator.alloc(DType::I64, {kLinearTopK, producer_groups, columns}, 256);
    out.group_keys   = allocator.alloc(DType::I64, {kLinearTopK, merge_groups, columns}, 256);
    if (secondary_groups != 0) {
        out.secondary_keys =
            allocator.alloc(DType::I64, {kLinearTopK, secondary_groups, columns}, 256);
    }
    out.group_done        = allocator.alloc(DType::I32, {columns}, 256);
    out.producer_groups   = producer_groups;
    out.merge_groups      = merge_groups;
    out.secondary_groups  = secondary_groups;
    out.columns           = columns;
    out.rows_per_producer = rows_per_producer;
    return out;
}

} // namespace ninfer::ops::detail
