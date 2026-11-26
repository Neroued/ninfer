#include "ops/softmax_attention/dense/context/plan.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {

ContextAttentionPlan select_context_attention_plan(AttentionHeadGeometry geometry,
                                                   ContextBucket bucket, std::int32_t batch,
                                                   std::int32_t tokens) {
    ContextBucketRange{bucket}.validate(kContextAttentionMaxBucket, "context_softmax_attention");
    if (tokens < 1 || tokens > 16 || batch < 1) {
        throw std::invalid_argument(
            "context_softmax_attention plan: T must be 1..16 and B must be positive");
    }
    int key_block = tokens > 8 && bucket >= ContextBucket::K128 ? 64 : 32;
    // Bucket capacities always cover at least this many tiles. Live lengths gate active splits.
    int splits           = bucket >= ContextBucket::K256 ? (tokens <= 8 ? 64 : 40) : 32;
    auto group3_schedule = Group3Schedule::Standard;
    int reduce_warps     = 4;
    const bool group3    = geometry.query_heads == 24;
    if (group3) {
        reduce_warps         = batch == 1 ? 1 : batch <= 8 ? 2 : 4;
        const bool wide_long = batch == 1 && tokens > 8 && bucket >= ContextBucket::K128;
        key_block            = 32;
        if (batch == 1) {
            splits = bucket == ContextBucket::K2 || wide_long ? 16 : 32;
        } else if (batch <= 8) {
            splits = std::max(8, (32 + batch - 1) / batch);
        } else {
            splits = static_cast<int>((64 + std::int64_t(batch) - 1) / batch);
        }
        if (tokens == 8 && splits > 1) {
            group3_schedule = batch == 1 && bucket <= ContextBucket::K8
                                  ? Group3Schedule::NarrowKeys
                                  : Group3Schedule::ParallelKeys;
        }
    } else if (batch > 8) {
        splits = static_cast<int>((std::int64_t(splits) * 8 + batch - 1) / batch);
    }
    return {
        .geometry  = geometry,
        .route     = splits == 1 ? ContextAttentionRoute::Direct : ContextAttentionRoute::SplitKv,
        .tokens    = tokens,
        .key_block = splits == 1 ? 32 : key_block,
        .split_capacity  = splits,
        .reduce_warps    = reduce_warps,
        .group3_schedule = group3_schedule,
    };
}

} // namespace ninfer::ops::detail
