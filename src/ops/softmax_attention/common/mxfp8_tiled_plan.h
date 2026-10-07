#pragma once

#include "ops/softmax_attention/common/causal_operands.h"
#include "ops/softmax_attention/common/causal_partition.h"
#include <algorithm>

namespace ninfer::ops::detail {

inline constexpr int kMxfp8TiledQueryRows = 128;
inline constexpr int kMxfp8TiledMaxSplits = kCausalTiledMaxSplits;

inline CausalKvPartition mxfp8_tiled_partition(int heads, int width, int visible_capacity,
                                               int multiprocessor_count) {
    const std::int64_t tiles =
        (static_cast<std::int64_t>(width) + kMxfp8TiledQueryRows - 1) / kMxfp8TiledQueryRows;
    return make_causal_tiled_partition(heads * tiles, visible_capacity, multiprocessor_count);
}

inline std::size_t mxfp8_tiled_workspace_bytes(int heads, int min_width, int max_width,
                                               int visible_capacity, int multiprocessor_count) {
    std::size_t maximum = 0;
    // A query-tile interval has one split target and increasing partial storage.
    // Check each interval's last width; checking max_width alone would miss a
    // larger allocation immediately before the split target decreases.
    for (std::int64_t begin = std::max(min_width, 17); begin <= max_width;) {
        const auto last =
            ((begin + kMxfp8TiledQueryRows - 1) / kMxfp8TiledQueryRows) * kMxfp8TiledQueryRows;
        const int end = static_cast<int>(std::min<std::int64_t>(max_width, last));
        const auto partition =
            mxfp8_tiled_partition(heads, end, visible_capacity, multiprocessor_count);
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, end, partition.capacity, 1);
        maximum = std::max(maximum, layout.peak_bytes(1));
        begin   = static_cast<std::int64_t>(end) + 1;
    }
    return maximum;
}

} // namespace ninfer::ops::detail
