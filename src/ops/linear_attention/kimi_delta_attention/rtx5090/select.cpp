#include "ops/linear_attention/kimi_delta_attention/plan.h"
#include "ops/linear_attention/kimi_delta_attention/input_conv/launch.h"
#include <limits>

namespace ninfer::ops::detail::kimi_delta_attention {
int select_value_tile(int value_heads, int multiprocessor_count, int active_segments,
                      int total_chunks, int longest_chunks) {
    const auto head_groups = static_cast<std::int64_t>(value_heads) * active_segments;
    // Small grids need the narrowest stripe. Otherwise minimize SM waves weighted
    // by the measured per-wave cost of this mixed BF16/TF32 kernel on SM120a.
    // Wider stripes reduce replicated packet reads but retain more state per CTA.
    if (head_groups <= multiprocessor_count / 16) return 8;
    int best                    = 64;
    double best_work            = std::numeric_limits<double>::infinity();
    const double average_chunks = static_cast<double>(total_chunks) / active_segments;

    struct Choice {
        int tile, cost;
    };

    // Account for the longest CTA separately so a few short segments do not
    // widen stripes on the critical path of a much longer segment.
    for (const auto [tile, cost] : {Choice{64, 34}, Choice{32, 20}, Choice{16, 13}}) {
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

int recurrence_window_tokens(int, int tokens) { return tokens; }

int recurrence_window_peak(int, int, int last) { return last; }

int ragged_window_chunks(int, int chunks) { return chunks; }

int ragged_window_peak(int, int, int last) { return last; }

RecordSchedule select_record_schedule(int heads, int tokens, int batch) {
    if (heads == 32 && tokens >= 16 && batch >= 64) return {16, true, 16};
    return {4};
}

int select_fold_warps(int layers, int heads, int batch, int multiprocessors) {
    const auto groups = std::int64_t(layers) * heads * batch;
    // 16-row tiles fill short one-layer grids and give large streaming grids more CTAs
    // per SM. The middle range benefits from amortizing record stages over 32 rows.
    return 2 * groups >= multiprocessors && groups <= 16LL * multiprocessors ? 8 : 4;
}
} // namespace ninfer::ops::detail::kimi_delta_attention

namespace ninfer::ops::detail::kda_input {
std::size_t workspace_capacity(int batch, int first, int last, bool) {
    return last <= 8 && last * batch <= 32
               ? 0
               : Tensor::metadata(DType::FP32, {kProjectionFeatures, last, batch}).bytes();
}
} // namespace ninfer::ops::detail::kda_input
