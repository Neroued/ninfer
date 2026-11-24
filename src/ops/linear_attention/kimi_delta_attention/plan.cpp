#include "ops/linear_attention/kimi_delta_attention/plan.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail::kimi_delta_attention {
RecurrenceRoute select_recurrence_route(int tokens) {
    return tokens < kFirstChunkedTokens ? RecurrenceRoute::Recurrent : RecurrenceRoute::Chunked;
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

ChunkWorkspace ragged_chunk_workspace(int value_heads, int total_chunks, int segments) {
    return chunk_workspace(value_heads, ragged_window_chunks(value_heads, total_chunks), segments);
}

std::size_t ragged_workspace_capacity(int heads, int min_tokens, int max_tokens, int min_segments,
                                      int max_segments) {
    constexpr std::int64_t short_limit        = kFirstChunkedTokens - 1;
    constexpr std::int64_t second_packet_cost = kChunkSize - short_limit;
    const std::int64_t first_s = min_segments, last_s = max_segments;
    const std::int64_t first_t = min_tokens, last_t = max_tokens;

    // With a single chunked segment, every other segment contributes 1..11 tokens.
    const int longest = int(last_t - first_s + 1);
    if (longest < kFirstChunkedTokens) return 0;
    const int shortest =
        int(std::max<std::int64_t>(kFirstChunkedTokens, first_t - short_limit * (last_s - 1)));
    auto bytes = recurrence_workspace_capacity(heads, shortest, longest);

    const auto max_long =
        std::min({last_s, last_t / kFirstChunkedTokens, (last_t - first_s) / short_limit});
    if (max_long < 2) return bytes;

    // For m long segments the fewest feasible total segments is max(m, first_s).
    // Buy a second packet in each long segment for five tokens before spending 16 on
    // further packets. This gives an attainable maximum at last_t.
    const auto max_packets = [&](std::int64_t m) {
        const auto spare   = last_t - std::max(m, first_s) - short_limit * m;
        const auto seconds = std::min(m, spare / second_packet_cost);
        return int(m + seconds + (spare - seconds * second_packet_cost) / kChunkSize);
    };
    const auto last_all_seconds =
        std::min({max_long, last_t / (kChunkSize + 1), (last_t - first_s) / kChunkSize});
    const auto consider = [&](std::int64_t m) {
        bytes = std::max(bytes, ragged_chunk_workspace(heads, max_packets(m), int(m)).bytes);
    };
    // Packet count increases while all second packets fit. After the first count where
    // they do not fit, each additional segment loses at least one packet. That packet
    // costs more than even the first spill of the compact index. Also consider max_long
    // for targets where the window cap holds packet storage flat while the index grows.
    const auto before = std::max<std::int64_t>(2, last_all_seconds);
    const auto after  = std::min(max_long, before + 1);
    consider(before);
    consider(after);
    consider(max_long);

    // The cheapest way to fit a small packet count is with two long segments and
    // as many short ones as the envelope permits. Feasible packet intervals for
    // adjacent segment counts overlap or touch.
    const auto most_short = std::min(last_s, last_t - 2 * short_limit) - 2;
    const auto long_tokens =
        std::max<std::int64_t>(2 * kFirstChunkedTokens, first_t - short_limit * most_short);
    const int least_packets = chunk_count(int(long_tokens));
    const int most_packets  = std::max(max_packets(before), max_packets(after));
    const int peak          = ragged_window_peak(heads, least_packets, most_packets);
    if (peak > ragged_window_chunks(heads, most_packets)) {
        bytes = std::max(bytes, ragged_chunk_workspace(heads, peak, 2).bytes);
    }
    return bytes;
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
