#pragma once

#include "ops/linear_attention/kimi_delta_attention/chunked/workspace.h"

namespace ninfer::ops::detail::kimi_delta_attention {
enum class RecurrenceRoute { Recurrent, Chunked };

struct RecurrencePlan {
    RecurrenceRoute route;
    int value_tile;
    std::size_t workspace_bytes;
    int window_tokens;
};

RecurrenceRoute select_recurrence_route(int tokens);
int select_value_tile(int value_heads, int multiprocessor_count);
int recurrence_window_tokens(int value_heads, int tokens);
int recurrence_window_peak(int value_heads, int first, int last);
int select_record_rows(int value_heads, int tokens, int batch);

struct FoldSchedule {
    int warps, values;
};

FoldSchedule select_fold_schedule(int layers, int value_heads, int width, int batch, int sms);
std::size_t recurrence_workspace_bytes(int value_heads, int tokens);
std::size_t recurrence_workspace_capacity(int value_heads, int first, int last);
RecurrencePlan select_recurrence_plan(int value_heads, int tokens, int sm_count);
} // namespace ninfer::ops::detail::kimi_delta_attention
