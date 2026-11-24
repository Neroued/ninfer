#include "ops/linear_attention/kimi_delta_attention/plan.h"
#include "ops/linear_attention/kimi_delta_attention/input_conv/launch.h"
#include <climits>

namespace ninfer::ops::detail::kimi_delta_attention {
int select_value_tile(int value_heads, int multiprocessor_count) {
    // Small grids need the narrowest stripe. Otherwise minimize SM waves weighted
    // by the per-wave cost of the mixed BF16/TF32 contraction.
    // The GLM local geometry selects 32 values per CTA.
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

int select_record_rows(int heads, int tokens, int batch) {
    return heads == 32 && tokens >= 12 && batch != 2 ? 8 : 4;
}

FoldSchedule select_fold_schedule(int layers, int heads, int width, int batch, int sms) {
    if (std::int64_t(layers) * heads * batch < 2LL * sms) return {4, 4};
    return width <= 2 ? FoldSchedule{16, 2} : FoldSchedule{4, 8};
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
