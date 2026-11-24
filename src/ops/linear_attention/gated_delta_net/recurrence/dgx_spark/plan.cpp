#include "ops/linear_attention/gated_delta_net/recurrence/plan.h"

namespace ninfer::ops::detail::gated_delta_net {
int first_chunked_tokens(int qk_heads, int value_heads) {
    return qk_heads == 8 && value_heads == 24 ? 11 : 16;
}

int max_packed_window_chunks(int qk_heads, int value_heads) {
    // Bound packet traffic within the 24 MiB L2 and leave room for inputs and state.
    constexpr int default_chunks = 64;
    static_assert(default_chunks <= chunked::kInlineSeqs);
    if (qk_heads == 16 && value_heads == 32) return 48;
    if (qk_heads == 16 && value_heads == 48) return 32;
    if (qk_heads == 8 && value_heads == 24) return 40;
    return default_chunks;
}

int select_ragged_value_tile(int qk_heads, int value_heads, int tokens, int segments,
                             int total_chunks, int multiprocessor_count) {
    if (segments == 1) {
        if (qk_heads == 8 && value_heads == 24) return tokens <= 128 ? 32 : 64;
        if (value_heads == 32 || value_heads == 48) {
            if (tokens <= 128) return 16;
            return value_heads == 48 ? 64 : 32;
        }
    }
    return select_value_tile_by_waves(value_heads, tokens, segments, total_chunks,
                                      multiprocessor_count);
}

} // namespace ninfer::ops::detail::gated_delta_net
