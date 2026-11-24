#include "ops/linear_attention/gated_delta_net/recurrence/plan.h"

#include <climits>

namespace ninfer::ops::detail::gated_delta_net {
int first_chunked_tokens(int /*qk_heads*/, int /*value_heads*/) { return 16; }

int max_packed_window_chunks(int /*qk_heads*/, int /*value_heads*/) { return INT_MAX; }

int select_ragged_value_tile(int /*qk_heads*/, int value_heads, int tokens, int segments,
                             int total_chunks, int multiprocessor_count) {
    return select_value_tile_by_waves(value_heads, tokens, segments, total_chunks,
                                      multiprocessor_count);
}

} // namespace ninfer::ops::detail::gated_delta_net
