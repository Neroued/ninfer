#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"

#include "ninfer/ops/silu_mul.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

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
    Q4LinearSwiGluScheduleId schedule;
};

constexpr Q4LinearSwiGluProblem kShape{34816, 17408, 5120, 5120, 1};

constexpr std::array<RouteSpec, 7> kRoutes{{
    {{1, 1}, Q4LinearSwiGluScheduleId::GemvPair},
    {{2, 128}, Q4LinearSwiGluScheduleId::Materialized},
    {{129, 256}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128},
    {{257, 384}, Q4LinearSwiGluScheduleId::Materialized},
    {{385, 512}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128},
    {{513, 640}, Q4LinearSwiGluScheduleId::Materialized},
    {{641, kQ4LinearSwiGluMaxCols}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128},
}};

constexpr bool catalog_is_closed() noexcept {
    std::int32_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.cols.first != expected || route.cols.last < route.cols.first) { return false; }
        expected = route.cols.last + 1;
    }
    return kRoutes.back().cols.last == kQ4LinearSwiGluMaxCols;
}

static_assert(catalog_is_closed(), "Q4 LinearSwiGLU routes must be exact, contiguous, and closed");

bool supported_shape(const Q4LinearSwiGluProblem& problem) noexcept {
    return problem.gate_up_rows == kShape.gate_up_rows &&
           problem.output_rows == kShape.output_rows && problem.k == kShape.k &&
           problem.padded_k == kShape.padded_k;
}

std::size_t checked_matrix_bytes(std::int32_t rows, std::int32_t cols) {
    const std::size_t r = static_cast<std::size_t>(rows);
    const std::size_t c = static_cast<std::size_t>(cols);
    if (c != 0 && r > std::numeric_limits<std::size_t>::max() / c) {
        throw std::overflow_error("q4 linear_swiglu: workspace element count overflows size_t");
    }
    const std::size_t elements = r * c;
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) {
        throw std::overflow_error("q4 linear_swiglu: workspace byte count overflows size_t");
    }
    return elements * sizeof(std::uint16_t);
}

Q4KernelVariant tiled_variant(Q4ScheduleId schedule, const Q4Problem& problem) {
    if (q4_candidate_is_legal(schedule, Q4KernelVariant::Full, problem)) {
        return Q4KernelVariant::Full;
    }
    if (q4_candidate_is_legal(schedule, Q4KernelVariant::Predicated, problem)) {
        return Q4KernelVariant::Predicated;
    }
    throw std::logic_error("q4 linear_swiglu: fixed materialized route is not physically legal");
}

Q4Plan materialized_plan(const Q4LinearSwiGluProblem& problem) {
    Q4ScheduleId schedule = Q4ScheduleId::MmaR64C128;
    if (problem.cols <= 4) {
        schedule = Q4ScheduleId::SimtR8C4;
    } else if (problem.cols <= 16) {
        schedule = Q4ScheduleId::SimtR8C8;
    }
    const Q4Problem base{problem.gate_up_rows, problem.k, problem.padded_k, problem.cols};
    return {schedule, tiled_variant(schedule, base)};
}

bool same_q4_plan(const std::optional<Q4Plan>& lhs, const std::optional<Q4Plan>& rhs) {
    if (lhs.has_value() != rhs.has_value()) { return false; }
    if (!lhs.has_value()) { return true; }
    return lhs->schedule == rhs->schedule && lhs->variant == rhs->variant;
}

} // namespace

const char* q4_linear_swiglu_schedule_name(Q4LinearSwiGluScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4LinearSwiGluScheduleId::GemvPair:
        return "linear_swiglu.q4.gemv.paired_rows";
    case Q4LinearSwiGluScheduleId::Materialized:
        return "linear_swiglu.q4.materialized";
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128:
        return "linear_swiglu.q4.mma.split_half_pair.r32.c128";
    }
    return "linear_swiglu.q4.unknown";
}

bool q4_linear_swiglu_admits(const Q4LinearSwiGluProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1 && problem.cols <= kQ4LinearSwiGluMaxCols;
}

Q4LinearSwiGluPlan q4_linear_swiglu_resolve_plan(const Q4LinearSwiGluProblem& problem) {
    if (!q4_linear_swiglu_admits(problem)) {
        throw std::invalid_argument(
            "q4 linear_swiglu: exact problem or column count is not admitted");
    }

    for (const RouteSpec& route : kRoutes) {
        if (!route.cols.contains(problem.cols)) { continue; }
        Q4LinearSwiGluPlan plan{
            route.schedule,
            Q4KernelVariant::None,
            std::nullopt,
            0,
            problem.cols <= kQ4LinearSwiGluQualifiedCols,
        };
        switch (route.schedule) {
        case Q4LinearSwiGluScheduleId::GemvPair:
            return plan;
        case Q4LinearSwiGluScheduleId::Materialized:
            plan.materialized_projection = materialized_plan(problem);
            plan.workspace_bytes         = checked_matrix_bytes(problem.gate_up_rows, problem.cols);
            return plan;
        case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128:
            plan.variant =
                (problem.cols % 128) == 0 ? Q4KernelVariant::Full : Q4KernelVariant::Predicated;
            return plan;
        }
    }
    throw std::logic_error("q4 linear_swiglu: admitted problem has no covering route");
}

std::size_t q4_linear_swiglu_capacity_workspace_bytes(std::int32_t gate_up_rows,
                                                      std::int32_t output_rows, std::int32_t k,
                                                      std::int32_t padded_k,
                                                      std::int32_t max_cols) {
    (void)q4_linear_swiglu_resolve_plan({gate_up_rows, output_rows, k, padded_k, max_cols});

    std::size_t maximum = 0;
    for (const RouteSpec& route : kRoutes) {
        if (route.cols.first > max_cols) { continue; }
        const std::int32_t endpoint = std::min(route.cols.last, max_cols);
        maximum                     = std::max(maximum, q4_linear_swiglu_resolve_plan(
                                        {gate_up_rows, output_rows, k, padded_k, endpoint})
                                                            .workspace_bytes);
    }
    return maximum;
}

void q4_linear_swiglu_execute_plan(const Q4LinearSwiGluPlan& plan, const Tensor& x, const Weight& w,
                                   Tensor& out, WorkspaceArena& ws, cudaStream_t stream) {
    const Q4LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q4LinearSwiGluPlan resolved = q4_linear_swiglu_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.variant != plan.variant ||
        !same_q4_plan(resolved.materialized_projection, plan.materialized_projection) ||
        resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("q4 linear_swiglu: plan does not match the exact problem");
    }

    switch (plan.schedule) {
    case Q4LinearSwiGluScheduleId::GemvPair:
        q4_linear_swiglu_gemv_pair_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::Materialized: {
        auto scratch_scope = ws.scope();
        Tensor gate_up     = ws.alloc(DType::BF16, {problem.gate_up_rows, problem.cols});
        q4_rowsplit_execute_plan(*plan.materialized_projection, x, w, gate_up, stream);
        silu_mul(gate_up.slice(0, 0, problem.output_rows),
                 gate_up.slice(0, problem.output_rows, problem.output_rows), out, stream);
        return;
    }
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128:
        q4_linear_swiglu_mma_split_half_pair_r32_c128_launch(plan.variant, x, w, out, stream);
        return;
    }
    throw std::logic_error("q4 linear_swiglu: unknown schedule");
}

void q4_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                               cudaStream_t stream) {
    const Q4LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q4LinearSwiGluPlan plan = q4_linear_swiglu_resolve_plan(problem);
    q4_linear_swiglu_execute_plan(plan, x, w, out, ws, stream);
}

} // namespace ninfer::ops::detail
