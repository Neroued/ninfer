#include "ops/softmax_attention/sliding_window/plan.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

void validate_extent(ContextBucket bucket, int batch, int tokens, const char* op) {
    ContextBucketRange{bucket}.validate(ContextBucket::M1, op);
    if (tokens < 1 || tokens > 16 || batch < 1)
        throw std::invalid_argument(std::string(op) + ": T must be 1..16 and B must be positive");
}

} // namespace

SlidingWindowLaunchShape select_sliding_window_attention_plan(AttentionHeadGeometry geometry,
                                                              std::uint32_t window,
                                                              ContextBucket bucket,
                                                              std::int32_t batch,
                                                              std::int32_t tokens) {
    constexpr const char* op = "sliding_window_attention";
    const bool full =
        geometry.query_heads == 32 && geometry.kv_heads == 8 && (window == 2048 || window == 4096);
    const bool half = geometry.query_heads == 16 && geometry.kv_heads == 4 && window == 2048;
    if (geometry.head_dim != 128 || (!full && !half))
        throw std::invalid_argument("sliding_window_attention: unsupported head/window profile");
    validate_extent(bucket, batch, tokens, op);
    // Keep enough independent KV-head CTAs for small batches, but limit repeated Q loads,
    // FP32 partial writes and merge traffic as the batch/query block grows.
    const bool one_reduce_warp =
        window == 4096 || (geometry.kv_heads == 4 && batch == 1 && tokens > 4 && tokens <= 8);
    int splits = kSlidingWindowMaxSplits;
    if (geometry.kv_heads == 4) {
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
    if (batch > 8) splits = static_cast<int>((std::int64_t(splits) * 8 + batch - 1) / batch);
    const auto bucket_tokens = context_bucket_tokens(bucket);
    return {
        .geometry       = geometry,
        .tokens         = tokens,
        .reduce_warps   = one_reduce_warp ? 1 : 4,
        .split_capacity = splits,
        .max_context    = bucket_tokens >= window ? std::numeric_limits<std::int32_t>::max()
                                                  : static_cast<std::int32_t>(bucket_tokens),
        .window         = static_cast<std::int32_t>(window),
    };
}

BlockSlidingWindowAttentionPlan
select_block_sliding_window_attention_plan(AttentionHeadGeometry geometry, std::uint32_t window,
                                           ContextBucket bucket, std::int32_t batch,
                                           std::int32_t tokens) {
    if (geometry.head_dim != 128 || geometry.query_heads != 24 || geometry.kv_heads != 8 ||
        window != 1024)
        throw std::invalid_argument(
            "block_sliding_window_attention: unsupported head/window profile");
    validate_extent(bucket, batch, tokens, "block_sliding_window_attention");

    const auto target_splits = (32 + std::int64_t(batch) - 1) / batch;
    int splits               = static_cast<int>(std::min<std::int64_t>(16, target_splits));
    auto schedule            = Group3Schedule::Standard;
    if (tokens == 8 && splits > 1) {
        // 17 splits distribute the 32 historical tiles and the query tile without
        // leaving a third tile on the longest split at a full ring.
        if (batch == 1) splits = 17;
        schedule = batch == 4                ? Group3Schedule::ParallelKeysStagedQuery
                   : batch <= 4 || batch > 8 ? Group3Schedule::ParallelKeys
                                             : Group3Schedule::NarrowKeys;
    }
    return {{geometry, tokens, batch == 1 ? 1 : 4, splits, std::numeric_limits<std::int32_t>::max(),
             static_cast<int>(window)},
            schedule};
}

} // namespace ninfer::ops::detail
