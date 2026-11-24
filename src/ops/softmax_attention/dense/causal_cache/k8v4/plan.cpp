#include "ops/softmax_attention/dense/causal_cache/k8v4/plan.h"
#include "ops/softmax_attention/kernels/causal/operands.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/instances.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kGroupedPrefillMaxWidth = 80;

CausalKvPartition select_tiled_partition(int heads, int width, int visible_capacity,
                                         int multiprocessor_count) {
    constexpr int rows = K8V4KvTiledInstance::kQueryRows;
    const auto tiles   = (std::int64_t(width) + rows - 1) / rows;
    return select_causal_tiled_partition(heads, tiles, visible_capacity, multiprocessor_count);
}

std::size_t tiled_workspace_bytes(int heads, int min_width, int max_width, int visible_capacity,
                                  int multiprocessor_count) {
    std::size_t maximum = 0;
    // A query-tile interval has one split target and increasing partial storage.
    // Check each interval's last width; checking max_width alone would miss a
    // larger allocation immediately before the split target decreases.
    for (std::int64_t begin = min_width; begin <= max_width;) {
        const auto last =
            ((begin + K8V4KvTiledInstance::kQueryRows - 1) / K8V4KvTiledInstance::kQueryRows) *
            K8V4KvTiledInstance::kQueryRows;
        const int end = static_cast<int>(std::min<std::int64_t>(max_width, last));
        const auto partition =
            select_tiled_partition(heads, end, visible_capacity, multiprocessor_count);
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, end, partition.capacity, 1);
        maximum = std::max(maximum, layout.peak_bytes(1));
        begin   = static_cast<std::int64_t>(end) + 1;
    }
    return maximum;
}

} // namespace

K8V4KvCausalPlan select_k8v4_kv_causal_plan(int heads, int width, int batch, ContextBucket bucket,
                                            int multiprocessor_count) {
    if (multiprocessor_count <= 0 || (heads != 24 && heads != 16) || width < 1 || batch < 1 ||
        (batch > 1 && width > 16))
        throw std::invalid_argument("K8V4 attention: invalid plan inputs");
    ContextBucketRange{bucket}.validate(ContextBucket::K256, "causal_softmax_attention");
    constexpr int grouped_limit = K8V4KvCausalPlan::kTokenTile;
    const auto family           = width <= grouped_limit             ? K8V4KvFamily::Grouped
                                  : width <= kGroupedPrefillMaxWidth ? K8V4KvFamily::ParallelGrouped
                                                                     : K8V4KvFamily::Tiled;
    if (family == K8V4KvFamily::Tiled)
        return {family,
                heads,
                width,
                batch,
                K8V4KvTiledInstance::kQueryRows,
                bucket,
                select_tiled_partition(heads, width, int(context_bucket_tokens(bucket)),
                                       multiprocessor_count)};
    const int tiles =
        family == K8V4KvFamily::ParallelGrouped ? (width + grouped_limit - 1) / grouped_limit : 1;
    const int query_tile        = family == K8V4KvFamily::ParallelGrouped && width <= 16
                                      ? (width + 1) / 2
                                      : std::min(width, grouped_limit);
    const int tiles_per_row     = (heads == 24 ? 4 : 2) * tiles;
    // Decode budgets two CTAs per SM; wider queries start with one and double the
    // budget if complete query-tile groups would leave over 10% of the SMs idle.
    const std::int64_t sms = multiprocessor_count;
    const auto split_target = [&](int query_tiles) {
        const auto budget =
            width == 1 || causal_query_tiles_underfill_sms(query_tiles, multiprocessor_count)
                ? 2 * sms
                : sms;
        return causal_partition_target(budget, query_tiles);
    };
    CausalKvPartition partition{1, split_target(batch * tiles_per_row)};
    partition.single_target = split_target(tiles_per_row);
    // Bound partial traffic by keeping enough KV work in each split.
    partition.key_shift = (width == 1 ? 7 : 8) - (heads == 16 ? 1 : 0);
    partition.capacity  = partition.active(int(context_bucket_tokens(bucket)));
    return {family, heads, width, batch, query_tile, bucket, partition};
}

std::size_t k8v4_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                    ContextBucket bucket, int multiprocessor_count) {
    std::size_t maximum = 0;
    for (int width = min_width; width <= std::min(max_width, kGroupedPrefillMaxWidth); ++width) {
        const auto plan =
            select_k8v4_kv_causal_plan(heads, width, batch, bucket, multiprocessor_count);
        if (plan.family == K8V4KvFamily::Tiled) continue;
        const int splits = plan.partition.capacity;
        if (splits == 1) continue;
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, width, splits, batch);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    return std::max(maximum, tiled_workspace_bytes(
                                 heads, std::max(min_width, kGroupedPrefillMaxWidth + 1), max_width,
                                 int(context_bucket_tokens(bucket)), multiprocessor_count));
}

kv::PagedWriteInstance select_k8v4_kv_write(int heads, int width, int batch) {
    using Instance = kv::PagedWriteInstance;
    if (batch > 1) return Instance::BatchRows256;
    return width >= 128 && heads == 2 ? Instance::Page8x256 : Instance::WarpRows256;
}

} // namespace ninfer::ops::detail
