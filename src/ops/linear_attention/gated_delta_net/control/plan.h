#pragma once

#include "ops/linear_attention/gated_delta_net/control/operands.h"
#include <limits>
#include <span>

namespace ninfer::ops::detail::gdn_control {

enum class Kernel { Gemv, TiledGemv, SmallSplit, Mma };
enum class RowMode { Paired, Merged };
enum class NormKernel { Simt, Mma, Materialized };

struct Plan {
    Kernel kernel          = Kernel::Gemv;
    std::int32_t tile_cols = 1, split_k = 1;
    RowMode rows = RowMode::Paired;
};

struct Route {
    std::int32_t first, last;
    Plan plan;
};

constexpr bool closed(std::span<const Route> routes) {
    std::int64_t next = 1;
    for (const auto& route : routes) {
        if (route.first != next || route.last < route.first) return false;
        next = std::int64_t(route.last) + 1;
    }
    return next == std::int64_t(std::numeric_limits<std::int32_t>::max()) + 1;
}

std::span<const Route> routes_n96_k2560();

struct NormPlan {
    NormKernel kernel;
    Plan projection;
    Plan fallback;
    int tile_cols = 0, threads = 0, capacity = 0;
};

Plan select_projection_plan(Geometry geometry, std::int32_t columns);
NormPlan select_norm_plan(Geometry geometry, std::int32_t columns);
std::size_t capacity(Geometry geometry, bool norm, std::int32_t first, std::int32_t last);

} // namespace ninfer::ops::detail::gdn_control
