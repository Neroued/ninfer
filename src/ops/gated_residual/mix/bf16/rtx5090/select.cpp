#include "ops/gated_residual/mix/bf16/select.h"
#include "ops/gated_residual/mix/bf16/workspace.h"
#include "ops/gated_residual/mix/bf16/implementation.h"
#include <algorithm>
#include <array>
#include <limits>
#include <span>

namespace ninfer::ops::detail::hc::mix::bf16 {
namespace {
struct Route {
    int first, last;
    Plan plan;
};

constexpr int kLast = std::numeric_limits<int>::max();

// TMA80 admits one resident CTA per SM. On 170 SMs, the compact 324/320-row Down
// crosses a full wave after 1200/1360 columns; partitioned K avoids that long tail.
constexpr std::array block_routes{
    Route{1, 1, {Down::Gemv, Up::Folded16x16}},
    Route{2, 4, {Down::Simt, Up::Folded16x16}},
    Route{5, 32, {Down::Split8x16, Up::Folded16x16}},
    Route{33, 64, {Down::Split8x32, Up::Folded32x32}},
    Route{65, 96, {Down::Split8x32, Up::TmaFolded32x64}},
    Route{97, 128, {Down::TmaSplit4, Up::TmaFolded32x64}},
    Route{129, 160, {Down::Split8x32, Up::TmaFolded32x64}},
    Route{161, 192, {Down::TmaSplit4, Up::TmaFolded16x64}},
    Route{193, 224, {Down::Split8x32, Up::TmaFolded32x64}},
    Route{225, 256, {Down::TmaSplit2, Up::TmaFolded16x64}},
    Route{257, 320, {Down::Tma96Split4, Up::TmaFolded16x64}},
    Route{321, 448, {Down::TmaSplit4, Up::TmaFolded16x64}},
    Route{449, 672, {Down::Tma96Split4, Up::TmaFolded16x64}},
    Route{673, 1200, {Down::Tma80, Up::TmaFolded16x64}},
    Route{1201, kLast, {Down::Tma96Split4, Up::TmaFolded16x64}},
};
constexpr std::array final_routes{
    Route{1, 1, {Down::Gemv, Up::Folded16x16}},
    Route{2, 4, {Down::Simt, Up::Folded16x16}},
    Route{5, 32, {Down::Split8x16, Up::Folded16x16}},
    Route{33, 64, {Down::Split8x32, Up::Folded32x32}},
    Route{65, 128, {Down::Split8x32, Up::TmaFolded32x64}},
    Route{129, 192, {Down::Tma96Split8, Up::TmaFolded16x64}},
    Route{193, 256, {Down::TmaSplit4, Up::TmaFolded32x64}},
    Route{257, 384, {Down::Tma96Split4, Up::TmaFolded16x64}},
    Route{385, 512, {Down::TmaSplit4, Up::TmaFolded16x64}},
    Route{513, 768, {Down::Tma96Split4, Up::TmaFolded16x64}},
    Route{769, 1360, {Down::Tma80, Up::TmaFolded16x64}},
    Route{1361, 1632, {Down::Tma96Split4, Up::TmaFolded16x64}},
    Route{1633, kLast, {Down::Tma96Split8, Up::TmaFolded16x64}},
};

constexpr bool closed(std::span<const Route> routes) {
    std::int64_t next = 1;
    for (const auto& route : routes) {
        if (route.first != next || route.last < route.first) return false;
        next = std::int64_t(route.last) + 1;
    }
    return next == std::int64_t(kLast) + 1;
}

static_assert(closed(block_routes) && closed(final_routes));

std::span<const Route> routes(Kind kind) {
    return kind == Kind::Block ? std::span<const Route>{block_routes}
                               : std::span<const Route>{final_routes};
}
} // namespace

Plan select_plan(Kind kind, std::int32_t columns) {
    validate_interval(columns, columns);
    for (const auto& route : routes(kind))
        if (columns >= route.first && columns <= route.last) return route.plan;
    throw std::logic_error("hc_mix: missing production route");
}

std::size_t capacity(Weights weights, std::int32_t first, std::int32_t last) {
    describe_weights(weights);
    validate_interval(first, last);
    std::size_t peak = 0;
    // Each route's allocation grows monotonically; the complete interval need not.
    for (const auto& route : routes(weights.kind)) {
        if (route.last < first || route.first > last) continue;
        peak = std::max(
            peak,
            layout(weights.kind, std::min(last, route.last), split_factor(route.plan.down)).bytes);
    }
    return peak;
}
} // namespace ninfer::ops::detail::hc::mix::bf16
