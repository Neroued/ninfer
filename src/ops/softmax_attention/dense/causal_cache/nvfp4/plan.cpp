#include "ops/softmax_attention/dense/causal_cache/nvfp4/plan.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/instances.h"
#include "ops/softmax_attention/kernels/causal/operands.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kGroupedPrefillMaxWidth = 192;
} // namespace

Nvfp4KvCausalPlan select_nvfp4_kv_causal_plan(int heads, int width, int batch, ContextBucket bucket,
                                              int multiprocessor_count) {
    if (multiprocessor_count <= 0 || (heads != 24 && heads != 16) || width < 1 || batch < 1 ||
        (batch > 1 && width > 16))
        throw std::invalid_argument("NVFP4 attention: invalid plan inputs");
    ContextBucketRange{bucket}.validate(ContextBucket::K256, "causal_softmax_attention");
    constexpr int grouped_limit = Nvfp4KvCausalPlan::kTokenTile;
    const auto family           = width <= grouped_limit ? Nvfp4KvFamily::Grouped
                                  : width <= kGroupedPrefillMaxWidth ? Nvfp4KvFamily::ParallelGrouped
                                                                     : Nvfp4KvFamily::Tiled;
    if (family == Nvfp4KvFamily::Tiled)
        return {family, heads, width, batch, Nvfp4KvTiledInstance::kQueryRows, bucket, {}};
    const int tiles =
        family == Nvfp4KvFamily::ParallelGrouped ? (width + grouped_limit - 1) / grouped_limit : 1;
    const int tiles_per_row     = (heads == 24 ? 4 : 2) * tiles;
    const int query_tile        = family == Nvfp4KvFamily::ParallelGrouped && width <= 16
                                      ? (width + 1) / 2
                                      : std::min(width, grouped_limit);
    const int row_tiles         = (query_tile * (heads == 24 ? 6 : 8) + 15) / 16;
    const std::int64_t sms      = multiprocessor_count;
    const auto split_target     = [&](int query_tiles) {
        const auto budget =
            row_tiles <= 2 || causal_query_tiles_underfill_sms(query_tiles, multiprocessor_count)
                    ? 2 * sms
                    : sms;
        return causal_partition_target(budget, query_tiles);
    };
    CausalKvPartition partition{1, split_target(batch * tiles_per_row)};
    partition.single_target = split_target(tiles_per_row);
    // Bound partial traffic by keeping enough KV work in each split.
    partition.key_shift = (row_tiles <= 2 ? 7 : 8) - (heads == 16 ? 1 : 0);
    partition.capacity  = partition.active(int(context_bucket_tokens(bucket)));
    return {family, heads, width, batch, query_tile, bucket, partition};
}

std::size_t nvfp4_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                     ContextBucket bucket, int multiprocessor_count) {
    std::size_t maximum = 0;
    for (int width = min_width; width <= std::min(max_width, kGroupedPrefillMaxWidth); ++width) {
        const auto plan =
            select_nvfp4_kv_causal_plan(heads, width, batch, bucket, multiprocessor_count);
        if (plan.family == Nvfp4KvFamily::Tiled) continue;
        const int splits = plan.partition.capacity;
        if (splits == 1) continue;
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
