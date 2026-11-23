#include "ops/softmax_attention/sliding_window/plan.h"

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
    const bool one_reduce_warp =
        window == 4096 || (geometry.kv_heads == 4 && batch == 1 && tokens > 4 && tokens <= 8);
    // Every supported window and bucket can hold at least 64 tiles of 32 context rows.
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
    if (batch > 8) { splits = static_cast<int>((std::int64_t(splits) * 8 + batch - 1) / batch); }
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

} // namespace ninfer::ops::detail
