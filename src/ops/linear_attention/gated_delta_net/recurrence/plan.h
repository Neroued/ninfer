#pragma once

#include "ops/linear_attention/gated_delta_net/recurrence/chunked/workspace.h"

namespace ninfer::ops::detail::gated_delta_net {
enum class RecurrenceRoute { Recurrent, Chunked };

struct RecurrencePlan {
    RecurrenceRoute route;
    int value_tile;
    chunked::WorkspaceLayout workspace;
};

int first_chunked_tokens(int qk_heads, int value_heads);
// A finite cap fits the inline index. Oversize segments run directly in their own windows.
int max_packed_window_chunks(int qk_heads, int value_heads);
int select_value_tile_by_waves(int value_heads, int longest, int segments, int total_chunks,
                               int sm_count);
int select_ragged_value_tile(int qk_heads, int value_heads, int longest, int segments,
                             int total_chunks, int sm_count);
std::size_t ragged_workspace_capacity(int qk_heads, int value_heads, int max_tokens,
                                      int min_segments, int max_segments);
RecurrenceRoute select_recurrence_route(int qk_heads, int value_heads, int tokens);
int select_value_tile(int qk_heads, int value_heads, int tokens, int multiprocessor_count);
chunked::WorkspaceLayout recurrence_workspace_layout(int qk_heads, int value_heads, int tokens);
RecurrencePlan select_recurrence_plan(int qk_heads, int value_heads, int tokens, int sm_count);
} // namespace ninfer::ops::detail::gated_delta_net
