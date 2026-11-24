#include "ops/softmax_attention/dense/causal_cache/nvfp4/plan.h"
#include "ops/softmax_attention/kernels/causal/operands.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
// Complete groups of independent query tiles should fill at least 90% of the SMs.
// Each dtype chooses whether a shortfall warrants a larger CTA budget.
inline constexpr bool causal_query_tiles_underfill_sms(int independent_tiles,
                                                       int multiprocessor_count) {
    const std::int64_t filled =
        (multiprocessor_count / independent_tiles) * static_cast<std::int64_t>(independent_tiles);
    return filled < static_cast<std::int64_t>(multiprocessor_count) * 9 / 10;
}

constexpr int kGroupedPrefillMaxWidth = 192;
} // namespace

Nvfp4KvCausalPlan select_nvfp4_kv_causal_plan(int heads, int width, int batch,
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
            select_nvfp4_kv_causal_plan(heads, width, batch, envelope, multiprocessor_count);
        if (plan.family == Nvfp4KvFamily::Tiled) continue;
        const int splits = plan.partition.capacity;
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, width, splits, batch);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    return maximum;
}

kv::PagedWriteInstance select_nvfp4_kv_write(int heads, int width, int batch) {
    using Instance = kv::PagedWriteInstance;
    if (batch > 1) return Instance::BatchRows256;
    return width >= 128 && heads == 2 ? Instance::Page8x256 : Instance::WarpRows256;
}

} // namespace ninfer::ops::detail
