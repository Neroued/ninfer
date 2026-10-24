#include "ops/linear_attention/kimi_delta_attention/plan.h"
#include "ops/linear_attention/kimi_delta_attention/input_conv/launch.h"
#include <climits>

namespace ninfer::ops::detail::kimi_delta_attention {
int select_value_tile(int value_heads, int multiprocessor_count) {
    // Small grids need the narrowest stripe. Otherwise minimize SM waves weighted
    // by the measured per-wave cost of this mixed BF16/TF32 kernel on SM120a.
    // Wider stripes reduce replicated packet reads but retain more state per CTA.
    if (static_cast<std::int64_t>(value_heads) * 16 <= multiprocessor_count) return 8;
    int best               = 64;
    std::int64_t best_work = INT64_MAX;

    struct Choice {
        int tile, cost;
    };

    for (const auto [tile, cost] : {Choice{64, 34}, Choice{32, 20}, Choice{16, 13}}) {
        const auto blocks = static_cast<std::int64_t>(value_heads) * (kStateDim / tile);
        const auto waves  = (blocks + multiprocessor_count - 1) / multiprocessor_count;
        const auto work   = waves * cost;
        if (work < best_work) {
            best      = tile;
            best_work = work;
        }
    }
    return best;
}

int recurrence_window_tokens(int, int tokens) { return tokens; }

int recurrence_window_peak(int, int, int last) { return last; }

int select_record_rows(int, int, int) { return 4; }

FoldSchedule select_fold_schedule(int layers, int heads, int width, int batch, int sms) {
    if (std::int64_t(layers) * heads * batch < 2LL * sms) return {4, 4};
    return width <= 8 ? FoldSchedule{16, 4} : FoldSchedule{8, 4};
}
} // namespace ninfer::ops::detail::kimi_delta_attention

namespace ninfer::ops::detail::kda_input {
std::size_t workspace_capacity(int batch, int first, int last, bool) {
    return last <= 8 && last * batch <= 32
               ? 0
               : Tensor::metadata(DType::FP32, {kProjectionFeatures, last, batch}).bytes();
}
} // namespace ninfer::ops::detail::kda_input
