#pragma once

#include "ops/linear_attention/kimi_delta_attention/chunked/workspace.h"

namespace ninfer::ops::detail::kimi_delta_attention {
enum class RecurrenceRoute { Recurrent, Chunked };
inline constexpr int kFirstChunkedTokens = 12;

struct RecurrencePlan {
    RecurrenceRoute route;
    int value_tile;
    std::size_t workspace_bytes;
    int window_tokens;
};

RecurrenceRoute select_recurrence_route(int tokens);
int select_value_tile(int value_heads, int multiprocessor_count, int active_segments = 1,
                      int total_chunks = 1, int longest_chunks = 1);
int recurrence_window_tokens(int value_heads, int tokens);
int recurrence_window_peak(int value_heads, int first, int last);
int ragged_window_chunks(int value_heads, int chunks);
int ragged_window_peak(int value_heads, int first, int last);

struct RecordSchedule {
    int rows;
    bool group_state_tiles = false;
    int steps              = 4;
};

RecordSchedule select_record_schedule(int value_heads, int tokens, int batch);

int select_fold_warps(int layers, int heads, int batch, int multiprocessors);

std::size_t recurrence_workspace_bytes(int value_heads, int tokens);
std::size_t recurrence_workspace_capacity(int value_heads, int first, int last);
std::size_t ragged_workspace_capacity(int value_heads, int min_tokens, int max_tokens,
                                      int min_segments, int max_segments);
ChunkWorkspace ragged_chunk_workspace(int value_heads, int total_chunks, int segments);
RecurrencePlan select_recurrence_plan(int value_heads, int tokens, int sm_count);
} // namespace ninfer::ops::detail::kimi_delta_attention
