#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

struct ColsSet {
    std::int32_t first;
    std::int32_t last;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last;
    }
};

struct RouteSpec {
    ColsSet cols;
    Bf16GdnGatingScheduleId schedule;
    std::int32_t split_k;
};

constexpr std::array<RouteSpec, 6> kRoutes{{
    {{1, 1}, Bf16GdnGatingScheduleId::GemvPairedRows, 1},
    {{2, 8}, Bf16GdnGatingScheduleId::SmallTSplit10, 10},
    {{9, 1024}, Bf16GdnGatingScheduleId::MmaCooperativeSplit8, 8},
    {{1025, 2048}, Bf16GdnGatingScheduleId::MmaCooperativeSplit4, 4},
    {{2049, 4096}, Bf16GdnGatingScheduleId::MmaCooperativeSplit2, 2},
    {{4097, kBf16GdnGatingMaxCols}, Bf16GdnGatingScheduleId::MmaUnsplit, 1},
}};

constexpr bool catalog_is_closed() noexcept {
    std::int32_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.cols.first != expected || route.cols.last < route.cols.first ||
            route.split_k <= 0) {
            return false;
        }
        expected = route.cols.last + 1;
    }
    return kRoutes.back().cols.last == kBf16GdnGatingMaxCols;
}

static_assert(catalog_is_closed(), "BF16 GDN gating routes must be exact and closed");

std::size_t checked_partial_bytes(std::int32_t split_k, std::int32_t cols) {
    constexpr std::size_t logical_rows = 96;
    const std::size_t split            = static_cast<std::size_t>(split_k);
    const std::size_t tokens           = static_cast<std::size_t>(cols);
    if (tokens > std::numeric_limits<std::size_t>::max() / logical_rows ||
        split > std::numeric_limits<std::size_t>::max() / (tokens * logical_rows)) {
        throw std::overflow_error("BF16 GDN gating workspace element count overflows size_t");
    }
    const std::size_t elements = split * tokens * logical_rows;
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::overflow_error("BF16 GDN gating workspace byte count overflows size_t");
    }
    return elements * sizeof(float);
}

} // namespace

const char* bf16_gdn_gating_schedule_name(Bf16GdnGatingScheduleId schedule) noexcept {
    switch (schedule) {
    case Bf16GdnGatingScheduleId::GemvPairedRows:
        return "gdn_gating_proj.bf16.gemv.paired_rows";
    case Bf16GdnGatingScheduleId::SmallTSplit10:
        return "gdn_gating_proj.bf16.small_t.split10";
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit8:
        return "gdn_gating_proj.bf16.mma.cooperative_split8";
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit4:
        return "gdn_gating_proj.bf16.mma.cooperative_split4";
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit2:
        return "gdn_gating_proj.bf16.mma.cooperative_split2";
    case Bf16GdnGatingScheduleId::MmaUnsplit:
        return "gdn_gating_proj.bf16.mma.unsplit";
    }
    return "gdn_gating_proj.bf16.unknown";
}

bool bf16_gdn_gating_admits(const Bf16GdnGatingProblem& problem) noexcept {
    return problem.heads == 48 && problem.input_rows == 5120 && problem.cols >= 1 &&
           problem.cols <= kBf16GdnGatingMaxCols;
}

Bf16GdnGatingPlan bf16_gdn_gating_resolve_plan(const Bf16GdnGatingProblem& problem) {
    if (!bf16_gdn_gating_admits(problem)) {
        throw std::invalid_argument(
            "BF16 GDN gating: exact problem or column count is not admitted");
    }

    for (const RouteSpec& route : kRoutes) {
        if (!route.cols.contains(problem.cols)) { continue; }
        const bool mma = route.schedule == Bf16GdnGatingScheduleId::MmaCooperativeSplit8 ||
                         route.schedule == Bf16GdnGatingScheduleId::MmaCooperativeSplit4 ||
                         route.schedule == Bf16GdnGatingScheduleId::MmaCooperativeSplit2 ||
                         route.schedule == Bf16GdnGatingScheduleId::MmaUnsplit;
        const Bf16GdnGatingTokenVariant variant =
            !mma ? Bf16GdnGatingTokenVariant::None
                 : ((problem.cols % 128) == 0 ? Bf16GdnGatingTokenVariant::Full
                                              : Bf16GdnGatingTokenVariant::Predicated);
        const std::size_t workspace =
            route.split_k > 1 ? checked_partial_bytes(route.split_k, problem.cols) : 0;
        return {route.schedule, variant, workspace, problem.cols <= kBf16GdnGatingQualifiedCols};
    }
    throw std::logic_error("BF16 GDN gating: admitted problem has no covering route");
}

std::size_t bf16_gdn_gating_capacity_workspace_bytes(std::int32_t max_cols) {
    (void)bf16_gdn_gating_resolve_plan({48, 5120, max_cols});
    std::size_t maximum = 0;
    for (const RouteSpec& route : kRoutes) {
        if (route.cols.first > max_cols) { continue; }
        const std::int32_t endpoint = std::min(route.cols.last, max_cols);
        maximum =
            std::max(maximum, bf16_gdn_gating_resolve_plan({48, 5120, endpoint}).workspace_bytes);
    }
    return maximum;
}

void bf16_gdn_gating_execute_plan(const Bf16GdnGatingPlan& plan, const Tensor& x,
                                  const Weight& a_weight, const Weight& b_weight,
                                  const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws,
                                  Tensor& g, Tensor& beta, cudaStream_t stream) {
    const Bf16GdnGatingPlan resolved = bf16_gdn_gating_resolve_plan({g.ne[0], x.ne[0], x.ne[1]});
    if (resolved.schedule != plan.schedule || resolved.token_variant != plan.token_variant ||
        resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("BF16 GDN gating: plan does not match the exact problem");
    }

    auto scratch_scope = ws.scope();
    DeviceSpan scratch{};
    if (plan.workspace_bytes != 0) { scratch = ws.alloc_bytes(plan.workspace_bytes); }

    switch (plan.schedule) {
    case Bf16GdnGatingScheduleId::GemvPairedRows:
        bf16_gdn_gating_proj_gemv_launch(x, a_weight, b_weight, A_log, dt_bias, g, beta, stream);
        return;
    case Bf16GdnGatingScheduleId::SmallTSplit10:
        bf16_gdn_gating_proj_small_t_split10_launch(x, a_weight, b_weight, A_log, dt_bias,
                                                    scratch.data, scratch.bytes, g, beta, stream);
        return;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit8:
        bf16_gdn_gating_proj_mma_split8_launch(plan.token_variant, x, a_weight, b_weight, A_log,
                                               dt_bias, scratch.data, g, beta, stream);
        return;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit4:
        bf16_gdn_gating_proj_mma_split4_launch(plan.token_variant, x, a_weight, b_weight, A_log,
                                               dt_bias, scratch.data, g, beta, stream);
        return;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit2:
        bf16_gdn_gating_proj_mma_split2_launch(plan.token_variant, x, a_weight, b_weight, A_log,
                                               dt_bias, scratch.data, g, beta, stream);
        return;
    case Bf16GdnGatingScheduleId::MmaUnsplit:
        bf16_gdn_gating_proj_mma_unsplit_launch(plan.token_variant, x, a_weight, b_weight, A_log,
                                                dt_bias, g, beta, stream);
        return;
    }
    throw std::logic_error("BF16 GDN gating: unknown schedule");
}

void bf16_gdn_gating_dispatch(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                              const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws,
                              Tensor& g, Tensor& beta, cudaStream_t stream) {
    const Bf16GdnGatingPlan plan = bf16_gdn_gating_resolve_plan({g.ne[0], x.ne[0], x.ne[1]});
    bf16_gdn_gating_execute_plan(plan, x, a_weight, b_weight, A_log, dt_bias, ws, g, beta, stream);
}

} // namespace ninfer::ops::detail
