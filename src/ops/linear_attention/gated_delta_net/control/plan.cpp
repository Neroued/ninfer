#include "ops/linear_attention/gated_delta_net/control/plan.h"
#include "ops/linear_attention/gated_delta_net/control/workspace.h"
#include "core/platform.h"
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail::gdn_control {
namespace {

constexpr int kLast = std::numeric_limits<std::int32_t>::max();

constexpr Plan mma(int tile, int split, RowMode rows = RowMode::Paired) {
    return {Kernel::Mma, tile, split, rows};
}

constexpr std::array kN96K5120{
    Route{1, 1, {Kernel::Gemv}},    Route{2, 8, {Kernel::SmallSplit, 8, 10}},
    Route{9, 1024, mma(128, 8)},    Route{1025, 2048, mma(128, 4)},
    Route{2049, 4096, mma(128, 2)}, Route{4097, kLast, mma(128, 1)},
};
constexpr std::array kN64K2048{
    Route{1, 127, mma(64, 16)},    Route{128, 1024, mma(64, 8)},   Route{1025, 2048, mma(64, 4)},
    Route{2049, 4096, mma(64, 2)}, Route{4097, kLast, mma(64, 1)},
};

// Materialized norm+control owns its tuning independently of control projection.
constexpr std::array kNormN96K5120{
    Route{1, 1, {Kernel::Gemv}},    Route{2, 8, {Kernel::SmallSplit, 8, 10}},
    Route{9, 1024, mma(128, 8)},    Route{1025, 2048, mma(128, 4)},
    Route{2049, 4096, mma(128, 2)}, Route{4097, kLast, mma(128, 1)},
};
constexpr std::array kNormN64K2048{
    Route{1, 127, mma(64, 16)},    Route{128, 1024, mma(64, 8)},   Route{1025, 2048, mma(64, 4)},
    Route{2049, 4096, mma(64, 2)}, Route{4097, kLast, mma(64, 1)},
};


static_assert(closed(kN96K5120) && closed(kN64K2048) && closed(kNormN96K5120) &&
              closed(kNormN64K2048));

std::span<const Route> routes(Geometry geometry, bool norm = false) {
    if (norm) {
        if (geometry.heads == 48 && geometry.input_rows == 5120) return kNormN96K5120;
        if (geometry.heads == 32 && geometry.input_rows == 2048) return kNormN64K2048;
        throw std::invalid_argument("gdn_control: unsupported norm geometry");
    }
    if (geometry.heads == 48 && geometry.input_rows == 5120) return kN96K5120;
    if (geometry.heads == 32 && geometry.input_rows == 2048) return kN64K2048;
    if (geometry.heads == 48 && geometry.input_rows == 2560) return routes_n96_k2560();
#if NINFER_TARGET_SPARK
    if (geometry.heads == 24 && geometry.input_rows == 2560) return routes_n48_k2560();
#endif
    throw std::invalid_argument("gdn_control: unsupported geometry");
}

} // namespace

Plan select_projection_plan(Geometry geometry, std::int32_t columns) {
    if (columns <= 0) throw std::invalid_argument("gdn_control: nonpositive projection extent");
    for (const auto& route : routes(geometry))
        if (columns >= route.first && columns <= route.last) return route.plan;
    throw std::logic_error("gdn_control: missing production route");
}

NormPlan select_norm_plan(Geometry geometry, std::int32_t columns) {
    if (columns <= 0) throw std::invalid_argument("gdn_control: nonpositive projection extent");
    Plan projection;
    for (const auto& route : routes(geometry, true))
        if (columns >= route.first && columns <= route.last) {
            projection = route.plan;
            break;
        }
    if (geometry.input_rows == 5120 && columns <= 42) {
        if (columns <= 2) return {NormKernel::Simt, {}, {}, 1, 512};
        if (columns <= 14) return {NormKernel::Simt, {}, {}, 2, 512};
        if (columns <= 28) return {NormKernel::Simt, {}, {}, 2, 256};
        return {NormKernel::Simt, {}, {}, 4, 256};
    }
    if (geometry.input_rows == 2048 && columns <= 16) {
        const int capacity = columns <= 6 ? 6 : columns <= 8 ? 8 : columns <= 12 ? 12 : 16;
        return {NormKernel::Mma, mma(64, 32), projection, 0, 0, capacity};
    }
    return {NormKernel::Materialized, projection, {}};
}

std::size_t capacity(Geometry geometry, bool norm, std::int32_t first, std::int32_t last) {
    validate_interval(first, last);
    if (norm) (void)select_norm_plan(geometry, first);
    std::size_t maximum = 0;
    const auto at       = [&](int columns) {
        if (!norm)
            return workspace_bytes(geometry, select_projection_plan(geometry, columns), columns);
        const auto plan = select_norm_plan(geometry, columns);
        return workspace_bytes(geometry, plan.projection, columns, plan.kernel == NormKernel::Mma);
    };
    for (const auto& route : routes(geometry, norm)) {
        if (route.last < first || route.first > last) continue;
        maximum = std::max(maximum, at(std::min(route.last, last)));
    }
    // Fused normalization has its own short interval inside the projection route.
    if (norm && geometry.input_rows == 2048 && first <= 16)
        maximum = std::max(maximum, at(std::min(last, 16)));
    return maximum;
}

} // namespace ninfer::ops::detail::gdn_control
