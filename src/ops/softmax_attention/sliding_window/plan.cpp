#include "ops/softmax_attention/sliding_window/plan.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {

SlidingWindowAttentionPlan
select_sliding_window_attention_plan(std::uint32_t window, std::int32_t tokens, std::int32_t batch,
                                     SlidingWindowAttentionExecutionEnvelope envelope) {
    if (window != 2048 && window != 4096) {
        throw std::invalid_argument("sliding_window_attention plan: unsupported window");
    }
    if (tokens < 1 || tokens > 16 || batch < 1 || batch > 8) {
        throw std::invalid_argument(
            "sliding_window_attention plan: T must be 1..16 and B must be 1..8");
    }
    if (envelope.min_context > envelope.max_context) {
        throw std::invalid_argument("sliding_window_attention plan: invalid envelope");
    }
    // Keep enough independent KV-head CTAs for small batches, but limit repeated Q loads,
    // FP32 partial writes and merge traffic as the batch/query block grows.
    const int direct_context_limit = window == 2048 && tokens > 8 && batch >= 6 ? 128 : 96;
    const bool direct = envelope.max_context <= static_cast<std::uint32_t>(direct_context_limit);
    constexpr int key_block          = 32;
    const int reduce_warps           = window == 2048 ? 4 : 1;
    const std::uint32_t context_rows = std::min(envelope.max_context, window - 1u);
    const std::int32_t context_tiles = (context_rows + key_block - 1u) / key_block;
    int split_limit                  = 32;
    if (window == 2048) {
        if (tokens <= 4) {
            split_limit = batch <= 2 ? 32 : batch <= 6 ? 16 : 8;
        } else if (tokens <= 8) {
            split_limit = batch == 1 ? 32 : batch == 2 ? 16 : 8;
            if (batch >= 6) split_limit = std::min(split_limit, std::max(4, context_tiles / 4));
        } else {
            split_limit = batch <= 2 ? 16 : batch <= 5 ? 8 : 4;
            if (batch >= 2) {
                const int tiles_per_split = batch == 2 ? 2 : 4;
                split_limit = std::min(split_limit, std::max(4, context_tiles / tiles_per_split));
            }
        }
    }
    return {
        .route =
            direct ? SlidingWindowAttentionRoute::Direct : SlidingWindowAttentionRoute::SplitKv,
        .tokens         = tokens,
        .warps          = (tokens + 3) / 4,
        .key_block      = key_block,
        .reduce_warps   = reduce_warps,
        .split_capacity = direct ? 1 : std::min(split_limit, std::max(1, context_tiles)),
        .max_context    = static_cast<std::int32_t>(envelope.max_context),
        .window         = static_cast<std::int32_t>(window),
    };
}

} // namespace ninfer::ops::detail
