#include "ops/linear_attention/gated_delta_net/recurrence/plan.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ninfer::ops::detail::gated_delta_net {
RecurrenceRoute select_recurrence_route(int qk_heads, int value_heads, int tokens) {
    return tokens < first_chunked_tokens(qk_heads, value_heads) ? RecurrenceRoute::Recurrent
                                                                : RecurrenceRoute::Chunked;
}

int select_value_tile(int qk_heads, int value_heads, int tokens, int sm_count) {
    return select_ragged_value_tile(qk_heads, value_heads, tokens, 1, chunked::chunk_count(tokens),
                                    sm_count);
}

int select_value_tile_by_waves(int value_heads, int longest, int segments, int total_chunks,
                               int sm_count) {
    // Narrow stripes fill a small grid. Wider stripes reduce replicated packet reads when
    // enough segments run concurrently. Also score the longest segment's wave separately
    // so short CTAs cannot justify a wide tile for the long tail of a skewed batch.
    const auto head_groups = static_cast<std::int64_t>(segments) * value_heads;
    if (head_groups <= sm_count / 16) return 8;
    const int longest_chunks = chunked::chunk_count(longest);
    int best                 = 64;
    double best_work         = std::numeric_limits<double>::infinity();

    struct Choice {
        int tile, cost;
    };

    for (const auto [tile, cost] : {Choice{64, 34}, Choice{32, 20}, Choice{16, 13}}) {
        const int stripes       = kStateDim / tile;
        const double blocks     = static_cast<double>(head_groups) * stripes;
        const double waves      = std::ceil(blocks / sm_count);
        const double tail_waves = std::ceil(static_cast<double>(value_heads) * stripes / sm_count);
        const double work =
            std::max(waves * total_chunks / segments, tail_waves * longest_chunks) * cost;
        if (work < best_work) {
            best      = tile;
            best_work = work;
        }
    }
    return best;
}

chunked::WorkspaceLayout recurrence_workspace_layout(int qk_heads, int value_heads, int tokens) {
    return select_recurrence_route(qk_heads, value_heads, tokens) == RecurrenceRoute::Recurrent
               ? chunked::WorkspaceLayout{}
               : chunked::workspace_layout(qk_heads, value_heads, tokens);
}

std::size_t ragged_workspace_capacity(int qk_heads, int value_heads, int max_tokens,
                                      int min_segments, int max_segments) {
    const int first = first_chunked_tokens(qk_heads, value_heads);
    using chunked::kChunkSize;
    // The two production crossovers satisfy 2*(first-1) >= 3*(17-first).
    // This is used below to bound the maximum to three neighboring counts.
    const int extra_for_second = kChunkSize + 1 - first;
    const auto bytes           = [&](int chunks, int segments) {
        return chunked::packed_workspace_layout(qk_heads, value_heads, chunks, segments)
            .total_bytes;
    };

    const int cap = max_packed_window_chunks(qk_heads, value_heads);
    // One segment can take all but one token per other required segment. This packet
    // count is attainable even when the segment exceeds the cap and runs directly.
    const int max_direct_tokens = max_tokens - min_segments + 1;
    std::size_t peak =
        recurrence_workspace_layout(qk_heads, value_heads, max_direct_tokens).total_bytes;
    if (peak && chunked::chunk_count(max_direct_tokens) >= cap) return peak;
    const int min_multi = std::max(min_segments, 2);
    if (max_segments < min_multi || max_tokens < min_multi) return peak;

    // For C chunked segments, reserve first tokens for each, plus one token for each
    // required short segment. The minimum feasible total segment count is max(C,min_multi).
    // Any larger count only takes tokens away from the chunked segments, and the index
    // depends on C alone. Max_tokens therefore suffices for every token interval.
    const int max_chunked =
        std::min({max_segments, max_tokens / first, (max_tokens - min_multi) / (first - 1)});
    if (max_chunked == 0) return peak;

    // Buy each second packet for 17-first extra tokens before buying further packets
    // for 16 each. When every second packet fits, packets and index size are monotone
    // in C. Above that point, every two additions lose at least one packet, whose
    // storage exceeds even the first spill of the inline index.
    const int last_all_seconds = std::min(
        {max_chunked, (max_tokens - min_multi) / kChunkSize, max_tokens / (kChunkSize + 1)});
    for (int c = std::max(1, last_all_seconds); c <= std::min(max_chunked, last_all_seconds + 2);
         ++c) {
        const int remaining = max_tokens - std::max(c, min_multi) - (first - 1) * c;
        const int second    = std::min(c, remaining / extra_for_second);
        const int chunks    = c + second + (remaining - second * extra_for_second) / kChunkSize;
        // A partition with at least cap packets can fill a window exactly: keep the
        // shortest prefix reaching cap packets and trim its last segment to fit.
        // Put spare tokens outside the prefix. If it uses every segment, pack the
        // spare tokens into those packets up to 16 tokens per packet. When there are
        // more tokens than that, a prefix using one fewer segment can still make
        // cap packets, with at least one multi-packet segment and room for a suffix.
        peak = std::max(peak, bytes(std::min(chunks, cap), std::min(c, cap)));
    }
    return peak;
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
