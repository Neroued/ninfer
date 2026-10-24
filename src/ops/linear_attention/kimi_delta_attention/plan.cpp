#include "ops/linear_attention/kimi_delta_attention/plan.h"
#include <stdexcept>

namespace ninfer::ops::detail::kimi_delta_attention {
RecurrenceRoute select_recurrence_route(int tokens) {
    return tokens < 12 ? RecurrenceRoute::Recurrent : RecurrenceRoute::Chunked;
}

std::size_t recurrence_workspace_bytes(int value_heads, int tokens) {
    return select_recurrence_route(tokens) == RecurrenceRoute::Recurrent
               ? 0
               : chunked_workspace_bytes(value_heads,
                                         recurrence_window_tokens(value_heads, tokens));
}

std::size_t recurrence_workspace_capacity(int heads, int first, int last) {
    return select_recurrence_route(last) == RecurrenceRoute::Recurrent
               ? 0
               : chunked_workspace_bytes(heads, recurrence_window_peak(heads, first, last));
}

RecurrencePlan select_recurrence_plan(int value_heads, int tokens, int sm_count) {
    const auto route = select_recurrence_route(tokens);
    if (route == RecurrenceRoute::Chunked && sm_count <= 0)
        throw std::invalid_argument("kimi_delta_attention: positive multiprocessor count required");
    return {route, route == RecurrenceRoute::Chunked ? select_value_tile(value_heads, sm_count) : 0,
            recurrence_workspace_bytes(value_heads, tokens),
            recurrence_window_tokens(value_heads, tokens)};
}
} // namespace ninfer::ops::detail::kimi_delta_attention
