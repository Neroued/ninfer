#include "ops/linear_attention/gated_delta_net/recurrence/plan.h"
#include <climits>

namespace ninfer::ops::detail::gated_delta_net {
int select_value_tile(int qk_heads, int value_heads, int tokens, int multiprocessor_count) {
    if (qk_heads == 8 && value_heads == 24) return tokens <= 128 ? 32 : 64;
    if (value_heads == 32 || value_heads == 48) {
        if (tokens <= 128) return 16;
        return value_heads == 48 ? 64 : 32;
    }
    // Small grids need the narrowest stripe. Otherwise use fixed relative tile costs
    // to weight the SM-wave count. These are launch-heuristic weights, not measured latency.
    // Wider stripes reduce replicated packet reads but retain more state per CTA.
    if (static_cast<std::int64_t>(value_heads) * 16 <= multiprocessor_count) return 8;
    int best               = 64;
    std::int64_t best_work = INT64_MAX;

    struct Choice {
        int tile, cost;
    };

    for (const auto [tile, cost] : {Choice{64, 34}, Choice{32, 20}, Choice{16, 13}}) {
        const auto blocks = static_cast<std::int64_t>(value_heads) * (kStateDim / tile);
        const auto waves  = (blocks + multiprocessor_count - 1) / multiprocessor_count;
        const auto work   = waves * cost;
        if (work < best_work) {
            best      = tile;
            best_work = work;
        }
    }
    return best;
}

} // namespace ninfer::ops::detail::gated_delta_net
