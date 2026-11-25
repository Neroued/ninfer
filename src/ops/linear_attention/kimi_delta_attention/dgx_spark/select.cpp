#include "ops/linear_attention/kimi_delta_attention/plan.h"
#include "ops/linear_attention/kimi_delta_attention/input_conv/launch.h"
#include <limits>

namespace ninfer::ops::detail::kimi_delta_attention {
int select_value_tile(int value_heads, int multiprocessor_count, int active_segments,
                      int total_chunks, int longest_chunks) {
    const auto head_groups = static_cast<std::int64_t>(value_heads) * active_segments;
    // Small grids need the narrowest stripe. Otherwise minimize SM waves weighted
    // by the per-wave cost of the mixed BF16/TF32 contraction.
    if (head_groups <= multiprocessor_count / 16) return 8;
    int best                    = 64;
    double best_work            = std::numeric_limits<double>::infinity();
    const double average_chunks = static_cast<double>(total_chunks) / active_segments;

    struct Choice {
        int tile, cost;
    };

    // Account for the longest CTA separately so a few short segments do not
    // widen stripes on the critical path of a much longer segment.
    // On GB10, the 64-value tile's longer wave costs more on short, concurrent runs.
    const int wide_cost = value_heads == 32 && active_segments > 1 && longest_chunks <= 4 ? 42 : 34;
    for (const auto [tile, cost] : {Choice{64, wide_cost}, Choice{32, 20}, Choice{16, 13}}) {
        const int stripes = kStateDim / tile;
        const auto waves =
            common::div_up(head_groups * stripes, std::int64_t(multiprocessor_count));
        const auto tail_waves =
            common::div_up(std::int64_t(value_heads) * stripes, std::int64_t(multiprocessor_count));
        const double work =
            std::max(waves * average_chunks, tail_waves * double(longest_chunks)) * cost;
        if (work < best_work) {
            best      = tile;
            best_work = work;
        }
    }
    return best;
}

namespace {
constexpr int kPacketWindow = 128, kPacketWindowThreshold = 256;
}

int recurrence_window_tokens(int heads, int tokens) {
    // Bound producer/consumer packets so the next recurrence can read them from L2.
    return heads == 32 && tokens > kPacketWindowThreshold ? kPacketWindow : tokens;
}

int recurrence_window_peak(int heads, int first, int last) {
    if (heads != 32 || last <= kPacketWindowThreshold) return last;
    return first <= kPacketWindowThreshold ? kPacketWindowThreshold : kPacketWindow;
}

int ragged_window_chunks(int heads, int chunks) {
    return heads == 32 && chunks > kPacketWindowThreshold / kChunkSize ? kPacketWindow / kChunkSize
                                                                       : chunks;
}

int ragged_window_peak(int heads, int first, int last) {
    constexpr int threshold = kPacketWindowThreshold / kChunkSize;
    if (heads != 32 || last <= threshold) return last;
    return first <= threshold ? threshold : ragged_window_chunks(heads, last);
}

RecordSchedule select_record_schedule(int heads, int tokens, int batch) {
    // Keep repeated Q/K/gate reads adjacent to each State at larger batches.
    if (heads == 32 && batch >= 16)
        return tokens >= 16 ? RecordSchedule{16, true, 16} : RecordSchedule{8, true};
    if (heads == 32) {
        if (batch >= 2 && tokens <= 4) return {2};
        if (batch == 1 && (tokens == 2 || tokens >= 5)) return {8};
    }
    return {heads == 32 && tokens >= 12 && batch != 2 ? 8 : 4};
}

int select_fold_warps(int layers, int heads, int, int) {
    // Keep small single-layer folds close to the record producer. Amortize the
    // staged decay and key loads across more state rows in full 32-head grids.
    if (heads >= 32) return layers == 1 ? 16 : 8;
    return 4;
}
} // namespace ninfer::ops::detail::kimi_delta_attention

namespace ninfer::ops::detail::kda_input {
namespace {
std::size_t input_bytes(int batch, int tokens, bool update) {
    const auto columns = std::int64_t(tokens) * batch;
    if (tokens <= 8 && columns <= 32) return 0;
    if (columns <= 32) return std::size_t(columns) * kProjectionFeatures * sizeof(float);
    return tiled_workspace(tokens, batch, update).bytes;
}
} // namespace

std::size_t workspace_capacity(int batch, int first, int last, bool update) {
    auto bytes = std::max(input_bytes(batch, first, update), input_bytes(batch, last, update));
    const int boundary = std::min(last, 32 / batch);
    if (boundary >= first) bytes = std::max(bytes, input_bytes(batch, boundary, update));
    return bytes;
}
} // namespace ninfer::ops::detail::kda_input
