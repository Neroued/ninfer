#pragma once

#include "core/layout.h"
#include "core/weight.h"
#include <algorithm>
#include <array>
#include "ops/common/numeric/math.h"

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kLinearTopK                = 16;
inline constexpr std::int32_t kLinearTopKMaxChunkColumns = 2048;
inline constexpr std::int32_t kLinearTopKGroupedRows     = 128;
inline constexpr std::int32_t kLinearTopKDirectRows      = 16;
inline constexpr std::int32_t kLinearTopKMergeFanIn      = 32;
inline constexpr std::int32_t kLinearTopKFullRows        = 248320;
inline constexpr std::int32_t kLinearTopKFullValidRows   = 248077;
inline constexpr std::int32_t kLinearTopKMappedRows      = 131072;
inline constexpr std::int32_t kLinearTopKHidden          = 5120;

inline int linear_topk_chunk_columns(int remaining, int large_threshold) {
    return std::min(remaining, remaining <= large_threshold ? 128 : kLinearTopKMaxChunkColumns);
}

template <class F>
void for_each_linear_topk_extent(int first, int last, int large_threshold, F&& visit) {
    std::array<bool, kLinearTopKMaxChunkColumns + 1> seen{};
    const int period =
        large_threshold >= kLinearTopKMaxChunkColumns ? 128 : kLinearTopKMaxChunkColumns;
    if (last >= period) seen[period] = true;
    const auto end = std::min<std::int64_t>(last, std::int64_t(first) + period - 1);
    for (std::int64_t extent = first; extent <= end; ++extent) {
        int remaining = extent % period;
        while (remaining) {
            const int columns = linear_topk_chunk_columns(remaining, large_threshold);
            seen[columns]     = true;
            remaining -= columns;
        }
    }
    for (int columns = 1; columns <= period; ++columns)
        if (seen[columns]) visit(columns);
}

inline Weight linear_topk_rows(const Weight& source, int first, int count) {
    Weight out                    = source;
    const std::size_t code_bytes  = source.qtype == QType::BF16          ? source.k * 2ULL
                                    : source.qtype == QType::Q4_G64_FP16 ? source.k / 2ULL
                                                                         : source.k;
    const std::size_t scale_bytes = source.qtype == QType::BF16                  ? 0
                                    : source.qtype == QType::MXFP8               ? source.k / 32ULL
                                    : source.qtype == QType::FP8_E4M3FN_ROW_BF16 ? 2
                                    : source.qtype == QType::Q8_G32_FP16         ? source.k / 16ULL
                                                                                 : source.k / 32ULL;
    out.qdata = static_cast<const std::byte*>(source.qdata) + first * code_bytes;
    if (scale_bytes)
        out.scales = static_cast<const std::byte*>(source.scales) + first * scale_bytes;
    out.n = out.shape[0] = out.padded_shape[0] = count;
    return out;
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
    std::int32_t row_begin        = 0;
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
