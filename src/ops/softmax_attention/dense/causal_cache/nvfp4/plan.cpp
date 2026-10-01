#include "ops/softmax_attention/dense/causal_cache/nvfp4/plan.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/instances.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/operands.h"
#include "ops/softmax_attention/common/mxfp8_tiled_plan.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kGroupedPrefillMaxWidth = 192;

inline CausalKvPartition nvfp4_tiled_partition(int heads, int width, int visible_capacity,
                                               int multiprocessor_count) {
    constexpr int QueryRows = Nvfp4KvTiledInstance::kQueryRows;
    const std::int64_t tiles =
        (static_cast<std::int64_t>(width) + QueryRows - 1) / QueryRows;
    return make_causal_tiled_partition(heads * tiles, visible_capacity, multiprocessor_count);
}

inline std::size_t nvfp4_tiled_workspace_bytes(int heads, int min_width, int max_width,
                                               int visible_capacity, int multiprocessor_count) {
    constexpr int QueryRows = Nvfp4KvTiledInstance::kQueryRows;
    std::size_t maximum     = 0;
    // A query-tile interval has one split target and increasing partial storage.
    // Check each interval's last width; checking max_width alone would miss a
    // larger allocation immediately before the split target decreases.
    for (std::int64_t begin = min_width; begin <= max_width;) {
        const auto last =
            ((begin + QueryRows - 1) / QueryRows) * QueryRows;
        const int end = static_cast<int>(std::min<std::int64_t>(max_width, last));
        const auto partition =
            nvfp4_tiled_partition(heads, end, visible_capacity, multiprocessor_count);
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, end, partition.capacity, 1);
        maximum = std::max(maximum, layout.peak_bytes(1));
        begin   = static_cast<std::int64_t>(end) + 1;
    }
    return maximum;
}
} // namespace

Nvfp4KvCausalPlan make_nvfp4_kv_causal_plan(int heads, int width, int batch,
                                            CausalAttentionExecutionEnvelope envelope,
                                            int multiprocessor_count) {
    if (multiprocessor_count <= 0 || (heads != 24 && heads != 16) || width < 1 || batch < 1 ||
        batch > 8 || (batch > 1 && width > 16) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys)
        throw std::invalid_argument("NVFP4 attention: invalid plan inputs");
    constexpr int grouped_limit = Nvfp4KvCausalPlan::kTokenTile;
    const auto family           = width <= grouped_limit ? Nvfp4KvFamily::Grouped
                                  : width <= kGroupedPrefillMaxWidth ? Nvfp4KvFamily::ParallelGrouped
                                                                     : Nvfp4KvFamily::Tiled;
    const int tiles =
        family == Nvfp4KvFamily::ParallelGrouped ? (width + grouped_limit - 1) / grouped_limit : 1;
    const int independent_tiles = batch * (heads == 24 ? 4 : 2) * tiles;
    const int query_tile        = family == Nvfp4KvFamily::ParallelGrouped && width <= 16
                                      ? (width + 1) / 2
                                      : std::min(width, grouped_limit);
    if (family == Nvfp4KvFamily::Tiled)
        return {family, heads, width, batch, query_tile, envelope,
                nvfp4_tiled_partition(heads, width, envelope.max_visible_keys,
                                      multiprocessor_count)};
    const int row_tiles         = (query_tile * (heads == 24 ? 6 : 8) + 15) / 16;
    const std::int64_t sms      = multiprocessor_count;
    const auto budget =
        row_tiles <= 2 || causal_query_tiles_underfill_sms(independent_tiles, multiprocessor_count)
            ? 2 * sms
            : sms;
    CausalKvPartition partition{1, causal_partition_target(budget, independent_tiles)};
    // Bound partial traffic by keeping enough KV work in each split.
    partition.key_shift = (row_tiles <= 2 ? 7 : 8) - (heads == 16 ? 1 : 0);
    partition.capacity  = partition.active(envelope.max_visible_keys);
    return {family, heads, width, batch, query_tile, envelope, partition};
}

std::size_t nvfp4_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                     CausalAttentionExecutionEnvelope envelope,
                                     int multiprocessor_count) {
    std::size_t maximum = 0;
    for (int width = min_width; width <= std::min(max_width, kGroupedPrefillMaxWidth); ++width) {
        const auto plan =
            make_nvfp4_kv_causal_plan(heads, width, batch, envelope, multiprocessor_count);
        const int splits = plan.partition.capacity;
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, width, splits, batch);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    return std::max(maximum, nvfp4_tiled_workspace_bytes(
                                 heads, std::max(min_width, kGroupedPrefillMaxWidth + 1),
                                 max_width, envelope.max_visible_keys, multiprocessor_count));
}

} // namespace ninfer::ops::detail
