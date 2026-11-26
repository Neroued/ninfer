#include "ops/softmax_attention/sliding_window/plan.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {

SlidingWindowAttentionPlan select_sliding_window_attention_plan(AttentionHeadGeometry geometry,
                                                                std::uint32_t window,
                                                                ContextBucket bucket,
                                                                std::int32_t batch,
                                                                std::int32_t tokens) {
    ContextBucketRange{bucket}.validate(ContextBucket::M1, "sliding_window_attention");
    if (tokens < 1 || tokens > 16 || batch < 1) {
        throw std::invalid_argument(
            "sliding_window_attention plan: T must be 1..16 and B must be positive");
    }
    // Keep enough independent KV-head CTAs for small batches, but limit repeated Q loads,
    // FP32 partial writes and merge traffic as the batch/query block grows.
    const bool group3 = geometry.query_heads == 24;
    const bool one_reduce_warp =
        (group3 && batch == 1) || window == 4096 ||
        (geometry.kv_heads == 4 && batch == 1 && tokens > 4 && tokens <= 8);
    auto group3_schedule = Group3Schedule::Standard;
    int splits = kSlidingWindowMaxSplits;
    if (group3) {
        const auto target_splits = (32 + std::int64_t(batch) - 1) / batch;
        splits                   = static_cast<int>(std::min<std::int64_t>(16, target_splits));
        if (tokens == 8 && splits > 1) {
            // 17 splits distribute the 32 historical tiles and the query tile without
            // leaving a third tile on the longest split at a full ring.
            if (batch == 1) splits = 17;
            group3_schedule = batch == 4                ? Group3Schedule::ParallelKeysStagedQuery
                              : batch <= 4 || batch > 8 ? Group3Schedule::ParallelKeys
                                                        : Group3Schedule::NarrowKeys;
        }
    } else if (geometry.kv_heads == 4) {
        if (tokens <= 4) {
            splits = batch == 1 ? 32 : batch == 2 || batch >= 6 ? 16 : 8;
        } else if (tokens <= 8) {
            splits = batch == 1 ? 16 : 8;
        } else {
            splits = batch == 1 ? 16 : batch == 2 ? 8 : batch <= 4 ? 4 : 2;
        }
    } else if (window == 2048) {
        if (tokens <= 4) {
            splits = batch <= 2 ? 32 : batch <= 6 ? 16 : 8;
        } else if (tokens <= 8) {
            splits = batch == 1 ? 32 : batch == 2 ? 16 : 8;
        } else {
            splits = batch <= 2 ? 16 : batch <= 5 ? 8 : 4;
        }
    }
    if (!group3 && batch > 8)
        splits = static_cast<int>((std::int64_t(splits) * 8 + batch - 1) / batch);
    const auto bucket_tokens = context_bucket_tokens(bucket);
    return {
        .geometry        = geometry,
        .tokens          = tokens,
        .reduce_warps    = one_reduce_warp ? 1 : 4,
        .split_capacity  = splits,
        .max_context     = bucket_tokens >= window ? std::numeric_limits<std::int32_t>::max()
                                                   : static_cast<std::int32_t>(bucket_tokens),
        .window          = static_cast<std::int32_t>(window),
        .group3_schedule = group3_schedule,
    };
}

} // namespace ninfer::ops::detail
