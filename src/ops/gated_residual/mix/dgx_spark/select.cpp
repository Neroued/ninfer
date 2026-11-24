#include "ops/gated_residual/mix/select.h"
#include "ops/gated_residual/mix/workspace.h"
#include "ops/gated_residual/mix/implementation.h"
#include <algorithm>
#include <array>
#include <limits>
#include <span>

namespace ninfer::ops::detail::hc::mix {
namespace {
struct Route {
    int first, last;
    Plan plan;
};

constexpr int kLast = std::numeric_limits<int>::max();

// Direct variants share one activation allocation; their wave counts differ.
constexpr std::array block_routes{
    Route{1, 1, {Down::Gemv, Up::Gemv}},
    Route{2, 2, {Down::Simt, Up::Folded16x16}},
    Route{3, 4, {Down::Simt4, Up::Folded16x16}},
    Route{5, 24, {Down::Split8x16, Up::Folded16x16}},
    Route{25, 32, {Down::Split8x16, Up::TmaFolded16x64}},
    Route{33, 64, {Down::Split8x32, Up::TmaFolded16x64}},
    Route{65, 96, {Down::Split8x32, Up::TmaFolded32x64}},
    Route{97, 128, {Down::TmaSplit4, Up::TmaFolded16x64}},
    Route{129, 160, {Down::Split8x32, Up::TmaFolded32Warp64}},
    Route{161, 192, {Down::TmaSplit4, Up::TmaFolded16x64}},
    Route{193, 224, {Down::Split8x32, Up::TmaFolded32Warp64}},
    Route{225, 256, {Down::TmaSplit2, Up::TmaFolded32Warp64}},
    Route{257, 320, {Down::Tma96Split4, Up::TmaFolded32Warp64}},
    Route{321, 512, {Down::Direct64S6, Up::TmaFolded32Warp64}},
    Route{513, 896, {Down::Direct128S4, Up::TmaFolded32Warp64}},
    Route{897, 1024, {Down::Direct128S4, Up::TmaPrefetched32Warp64}},
    Route{1025, kLast, {Down::Direct64S6, Up::TmaPrefetched32Warp64}},
};

constexpr std::array final_routes{
    Route{1, 1, {Down::Gemv, Up::Gemv}},
    Route{2, 2, {Down::Simt, Up::Folded16x16}},
    Route{3, 4, {Down::Simt4, Up::Folded16x16}},
    Route{5, 24, {Down::Split8x16, Up::Folded16x16}},
    Route{25, 32, {Down::Split8x16, Up::TmaFolded16x64}},
    Route{33, 64, {Down::Split8x32, Up::TmaFolded16x64}},
    Route{65, 96, {Down::Split8x32, Up::TmaFolded32x64}},
    Route{97, 128, {Down::Split8x32, Up::TmaFolded16x64}},
    Route{129, 160, {Down::Tma96Split8, Up::TmaFolded32Warp64}},
    Route{161, 192, {Down::Tma96Split8, Up::TmaFolded16x64}},
    Route{193, 256, {Down::TmaSplit4, Up::TmaFolded32Warp64}},
    Route{257, 320, {Down::Tma96Split4, Up::TmaFolded32Warp64}},
    Route{321, 576, {Down::Direct64S6, Up::TmaFolded32Warp64}},
    Route{577, 768, {Down::Direct128S4, Up::TmaFolded32Warp64}},
    Route{769, 896, {Down::Direct64S3, Up::TmaFolded32Warp64}},
    Route{897, 1216, {Down::Direct64S3, Up::TmaPrefetched32Warp64}},
    Route{1217, kLast, {Down::Direct64S6, Up::TmaPrefetched32Warp64}},
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
    for (const auto& route : routes(kind)) {
        if (columns < route.first || columns > route.last) continue;
        auto plan              = route.plan;
        const int direct_start = kind == Kind::Block ? 1024 : 1216;
        if (columns > direct_start) {
            const std::int64_t tt = (std::int64_t(columns) + 63) / 64;
            const auto work64     = (((kind == Kind::Block ? 6 : 5) * tt + 47) / 48) * 64 * 64;
            const auto deep       = ((3 * tt + 47) / 48) * 128 * 64;
            const auto two        = ((3 * tt + 95) / 96) * 2 * 128 * 64;
            if (deep < two && deep <= work64)
                plan.down = Down::Direct128S4;
            else if (kind == Kind::Block ? 3 * two <= 4 * work64 : two <= work64)
                plan.down = Down::Direct128S2;
            else
                plan.down = Down::Direct64S6;
        }
        return plan;
    }
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
} // namespace ninfer::ops::detail::hc::mix
