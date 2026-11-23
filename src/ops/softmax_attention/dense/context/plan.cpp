#include "ops/softmax_attention/dense/context/plan.h"

#include <stdexcept>

namespace ninfer::ops::detail {

ContextAttentionPlan select_context_attention_plan(ContextBucket bucket, std::int32_t batch,
                                                   std::int32_t tokens) {
    ContextBucketRange{bucket}.validate(kContextAttentionMaxBucket, "context_softmax_attention");
    if (tokens < 1 || tokens > 16 || batch < 1) {
        throw std::invalid_argument(
            "context_softmax_attention plan: T must be 1..16 and B must be positive");
    }
    const int key_block = tokens > 8 && bucket >= ContextBucket::K128 ? 64 : 32;
    // Bucket capacities always cover at least this many tiles. Live lengths gate active splits.
    int splits = bucket == ContextBucket::K256 ? (tokens <= 8 ? 64 : 40) : 32;
    if (batch > 8) { splits = static_cast<int>((std::int64_t(splits) * 8 + batch - 1) / batch); }
    return {
        .route     = splits == 1 ? ContextAttentionRoute::Direct : ContextAttentionRoute::SplitKv,
        .tokens    = tokens,
        .key_block = splits == 1 ? 32 : key_block,
        .split_capacity = splits,
    };
}

} // namespace ninfer::ops::detail
