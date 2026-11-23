#include "ops/softmax_attention/dense/causal_cache/int8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/instances.h"
#include "ops/softmax_attention/kernels/causal/operands.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kGroupedPrefillMaxWidth = 256;
} // namespace

Int8KvCausalPlan select_int8_kv_causal_plan(int heads, int width, int batch, ContextBucket bucket,
                                            int multiprocessor_count) {
    if (multiprocessor_count <= 0 || (heads != 24 && heads != 16) || width < 1 || batch < 1 ||
        (batch > 1 && width > 16))
        throw std::invalid_argument("INT8 attention: invalid plan inputs");
    ContextBucketRange{bucket}.validate(ContextBucket::K256, "causal_softmax_attention");
    constexpr int grouped_limit = Int8KvCausalPlan::kTokenTile;
    const auto family           = width <= grouped_limit             ? Int8KvFamily::Grouped
                                  : width <= kGroupedPrefillMaxWidth ? Int8KvFamily::ParallelGrouped
                                                                     : Int8KvFamily::Tiled;
    if (family == Int8KvFamily::Tiled)
        return {family, heads, width, batch, Int8KvTiledInstance::kQueryRows, bucket, {}};
    const int tiles =
        family == Int8KvFamily::ParallelGrouped ? (width + grouped_limit - 1) / grouped_limit : 1;
    const int tiles_per_row = (heads == 24 ? 4 : 2) * tiles;
    const std::int64_t sms  = multiprocessor_count;
    const auto split_target = [&](int query_tiles) {
        const auto budget =
            heads == 24 || width <= 4 ||
                    causal_query_tiles_underfill_sms(query_tiles, multiprocessor_count)
                ? 2 * sms
                : sms;
        return causal_partition_target(budget, query_tiles);
    };
    CausalKvPartition partition{1, split_target(batch * tiles_per_row)};
    partition.single_target = split_target(tiles_per_row);
    // Bound partial traffic by keeping enough KV work in each split.
    partition.key_shift = (width == 1 ? 7 : 8) - (heads == 16 ? 1 : 0);
    partition.capacity  = partition.active(int(context_bucket_tokens(bucket)));
    return {family, heads, width, batch, std::min(width, grouped_limit), bucket, partition};
}

std::size_t int8_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                    ContextBucket bucket, int multiprocessor_count) {
    std::size_t maximum = 0;
    for (int width = min_width; width <= std::min(max_width, kGroupedPrefillMaxWidth); ++width) {
        const auto plan =
            select_int8_kv_causal_plan(heads, width, batch, bucket, multiprocessor_count);
        if (plan.family == Int8KvFamily::Tiled) continue;
        const int splits = plan.partition.capacity;
        if (splits == 1) continue;
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, width, splits, batch);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    return maximum;
}

kv::PagedWriteInstance select_int8_kv_write(int heads, int width, int batch) {
    using Instance = kv::PagedWriteInstance;
    if (batch > 1) return Instance::BatchRows256;
    return width >= 128 && heads == 2 ? Instance::Page8x256 : Instance::WarpRows256;
}

} // namespace ninfer::ops::detail
