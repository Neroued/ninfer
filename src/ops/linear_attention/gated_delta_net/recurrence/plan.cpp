#include "ops/linear_attention/gated_delta_net/recurrence/plan.h"

namespace ninfer::ops::detail::gated_delta_net {
chunked::WorkspaceLayout recurrence_workspace_layout(int qk_heads, int value_heads, int tokens) {
    return select_recurrence_route(qk_heads, value_heads, tokens) == RecurrenceRoute::Recurrent
               ? chunked::WorkspaceLayout{}
               : chunked::workspace_layout(qk_heads, value_heads, tokens);
}

RecurrencePlan select_recurrence_plan(int qk_heads, int value_heads, int tokens, int sm_count) {
    const auto route = select_recurrence_route(qk_heads, value_heads, tokens);
    return {route,
            route == RecurrenceRoute::Chunked
                ? select_value_tile(qk_heads, value_heads, tokens, sm_count)
                : 0,
            recurrence_workspace_layout(qk_heads, value_heads, tokens)};
}
} // namespace ninfer::ops::detail::gated_delta_net
