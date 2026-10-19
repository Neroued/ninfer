#pragma once

#include "ops/linear_attention/gated_delta_net/recurrence/chunked/workspace.h"

namespace ninfer::ops::detail::gated_delta_net {
enum class RecurrenceRoute { Recurrent, Chunked };

struct RecurrencePlan {
    RecurrenceRoute route;
    int value_tile;
    chunked::WorkspaceLayout workspace;
};

RecurrenceRoute select_recurrence_route(int qk_heads, int value_heads, int tokens);
int select_value_tile(int qk_heads, int value_heads, int tokens, int multiprocessor_count);
chunked::WorkspaceLayout recurrence_workspace_layout(int qk_heads, int value_heads, int tokens);
RecurrencePlan select_recurrence_plan(int qk_heads, int value_heads, int tokens, int sm_count);
} // namespace ninfer::ops::detail::gated_delta_net
