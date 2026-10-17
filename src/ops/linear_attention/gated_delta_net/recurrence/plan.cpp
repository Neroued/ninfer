#include "ops/linear_attention/gated_delta_net/recurrence/plan.h"
#include <stdexcept>

namespace ninfer::ops::detail::gated_delta_net {
RecurrenceRoute select_recurrence_route(int tokens) {
    return tokens < 16 ? RecurrenceRoute::Recurrent : RecurrenceRoute::Chunked;
}

chunked::WorkspaceLayout recurrence_workspace_layout(int qk_heads, int value_heads, int tokens) {
    return select_recurrence_route(tokens) == RecurrenceRoute::Recurrent
               ? chunked::WorkspaceLayout{}
               : chunked::workspace_layout(qk_heads, value_heads, tokens);
}

RecurrencePlan select_recurrence_plan(int qk_heads, int value_heads, int tokens, int sm_count) {
    const auto route = select_recurrence_route(tokens);
    return {route,
            route == RecurrenceRoute::Chunked ? select_value_tile(value_heads, tokens, sm_count)
                                              : 0,
            recurrence_workspace_layout(qk_heads, value_heads, tokens)};
}
} // namespace ninfer::ops::detail::gated_delta_net
