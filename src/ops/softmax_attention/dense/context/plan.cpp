#include "ops/softmax_attention/dense/context/plan.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {

ContextAttentionPlan select_context_attention_plan(std::int32_t tokens,
                                                   ContextAttentionExecutionEnvelope envelope) {
    if (tokens < 1 || tokens > 16) {
        throw std::invalid_argument("context_softmax_attention plan: T must be 1..16");
    }
    if (envelope.min_context > envelope.max_context) {
        throw std::invalid_argument("context_softmax_attention plan: invalid envelope");
    }
    const std::int32_t warps = (tokens + 3) / 4;
    const bool direct        = envelope.max_context == 0;
    const std::int32_t key_block =
        direct || tokens <= 8 || envelope.max_context <= 65536u ? 32 : 64;
    std::int32_t split_limit = 32;
    if (tokens <= 8) {
        split_limit =
            envelope.max_context <= 131072u ? 32 : (envelope.max_context <= 196608u ? 48 : 64);
    } else if (key_block == 64) {
        split_limit =
            envelope.max_context <= 131072u ? 32 : (envelope.max_context <= 196608u ? 38 : 40);
    }
    const std::uint32_t envelope_tiles =
        (envelope.max_context + static_cast<std::uint32_t>(key_block) - 1u) /
        static_cast<std::uint32_t>(key_block);
    const std::int32_t splits =
        direct ? 1 : std::min(split_limit, std::max(1, static_cast<std::int32_t>(envelope_tiles)));
    return {
        .route          = direct ? ContextAttentionRoute::Direct : ContextAttentionRoute::SplitKv,
        .tokens         = tokens,
        .warps          = warps,
        .key_block      = key_block,
        .split_capacity = splits,
    };
}

} // namespace ninfer::ops::detail
