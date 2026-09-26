#include "core/weight.h"
#include "ops/linear_add/q5/q5_linear_add_plan.h"

#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct ColsSet {
    std::int32_t first;
    std::int32_t last;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last;
    }
};

struct SupportSpec {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
};

struct RouteSpec {
    ColsSet cols;
    Q5LinearAddScheduleId schedule;
};

constexpr std::array<SupportSpec, 2> kSupports{{
    {5120, 6144, 6144},
    {5120, 17408, 17408},
}};

constexpr std::array<RouteSpec, 13> kK6144Routes{{
    {{1, 4}, Q5LinearAddScheduleId::Split2ExactResidual},
    {{5, 8}, Q5LinearAddScheduleId::SlicedR16T8W4S2},
    {{9, 16}, Q5LinearAddScheduleId::SlicedR16T16W4S2},
    {{17, 24}, Q5LinearAddScheduleId::SlicedR16T24W4S2},
    {{25, 32}, Q5LinearAddScheduleId::SlicedR32T32W4S2},
    {{33, 48}, Q5LinearAddScheduleId::SlicedR32T24W4S2Pairwise},
    {{49, 64}, Q5LinearAddScheduleId::SlicedR32T32W4S2},
    {{65, 96}, Q5LinearAddScheduleId::MmaResidualR32T32K128},
    {{97, 128}, Q5LinearAddScheduleId::SlicedR32T32W2S2},
    {{129, 160}, Q5LinearAddScheduleId::SlicedR32T64W2S1},
    {{161, 256}, Q5LinearAddScheduleId::MmaResidualR32T128},
    {{257, 512}, Q5LinearAddScheduleId::MmaResidualR64T128},
    {{513, kAnyCols}, Q5LinearAddScheduleId::MmaResidualR64T128Tail},
}};

constexpr std::array<RouteSpec, 13> kK17408Routes{{
    {{1, 4}, Q5LinearAddScheduleId::Split2ExactResidual},
    {{5, 8}, Q5LinearAddScheduleId::SlicedR16T8W4S2},
    {{9, 16}, Q5LinearAddScheduleId::SlicedR16T16W4S2},
    {{17, 24}, Q5LinearAddScheduleId::SlicedR16T24W4S2},
    {{25, 32}, Q5LinearAddScheduleId::SlicedR32T32W4S2},
    {{33, 48}, Q5LinearAddScheduleId::SlicedR32T24W4S2Pairwise},
    {{49, 64}, Q5LinearAddScheduleId::SlicedR32T32W4S1},
    {{65, 96}, Q5LinearAddScheduleId::SlicedR32T32W4S1},
    {{97, 128}, Q5LinearAddScheduleId::SlicedR32T32W2S2},
    {{129, 160}, Q5LinearAddScheduleId::SlicedR32T64W2S1},
    {{161, 256}, Q5LinearAddScheduleId::MmaResidualR32T128},
    {{257, 512}, Q5LinearAddScheduleId::MmaResidualR64T128},
    {{513, kAnyCols}, Q5LinearAddScheduleId::MmaResidualR64T128Tail},
}};

template <std::size_t N>
constexpr bool catalog_is_closed(const std::array<RouteSpec, N>& routes) noexcept {
    std::int64_t expected = 1;
    for (const RouteSpec& route : routes) {
        if (route.cols.first != expected || route.cols.last < route.cols.first) { return false; }
        expected = static_cast<std::int64_t>(route.cols.last) + 1;
    }
    return routes.back().cols.last == kAnyCols &&
           expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(kK6144Routes) && catalog_is_closed(kK17408Routes),
              "Q5 LinearAdd routes must be exact, contiguous, and closed");

bool supported_shape(const Q5LinearAddProblem& problem) noexcept {
    for (const SupportSpec& support : kSupports) {
        if (problem.rows == support.rows && problem.k == support.k &&
            problem.padded_k == support.padded_k) {
            return true;
        }
    }
    return false;
}

// Keep complete 512-token waves on the wide collective MMA route. A short
// remainder uses the measured narrow schedules to avoid another mostly empty wave.
constexpr std::int32_t kWaveCols       = 512;
constexpr std::int32_t kNarrowTailCols = 192;

void launch_wide_with_narrow_tail(const Tensor& x, const Weight& w, Tensor& residual_out,
                                  WorkspaceArena& ws, cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    const std::int32_t wide = (cols / kWaveCols) * kWaveCols;
    const std::int32_t tail = cols - wide;
    if (wide == 0 || tail == 0 || tail > kNarrowTailCols) {
        q5_linear_add_mma_r64_t128_launch(x, w, residual_out, stream);
        return;
    }

    const Tensor x_wide = x.slice(1, 0, wide);
    Tensor out_wide     = residual_out.slice(1, 0, wide);
    q5_linear_add_mma_r64_t128_launch(x_wide, w, out_wide, stream);

    const Tensor x_tail = x.slice(1, wide, tail);
    Tensor out_tail     = residual_out.slice(1, wide, tail);
    q5_linear_add_execute_plan(
        q5_linear_add_resolve_plan({residual_out.ne[0], x.ne[0], w.padded_shape[1], x_tail.ne[1]}),
        x_tail, w, out_tail, ws, stream);
}

} // namespace

const char* q5_linear_add_schedule_name(Q5LinearAddScheduleId schedule) noexcept {
    switch (schedule) {
    case Q5LinearAddScheduleId::Split2ExactResidual:
        return "linear_add.q5.simt.split2.exact.residual";
    case Q5LinearAddScheduleId::SlicedR16T8W4S2:
        return "linear_add.q5.sliced.r16.t8.w4.s2.residual";
    case Q5LinearAddScheduleId::SlicedR16T16W4S2:
        return "linear_add.q5.sliced.r16.t16.w4.s2.residual";
    case Q5LinearAddScheduleId::SlicedR16T24W4S2:
        return "linear_add.q5.sliced.r16.t24.w4.s2.residual";
    case Q5LinearAddScheduleId::SlicedR32T32W4S2:
        return "linear_add.q5.sliced.r32.t32.w4.s2.residual";
    case Q5LinearAddScheduleId::SlicedR32T24W4S2Pairwise:
        return "linear_add.q5.sliced.r32.t24.w4.s2.pairwise.residual";
    case Q5LinearAddScheduleId::SlicedR32T32W4S1:
        return "linear_add.q5.sliced.r32.t32.w4.s1.residual";
    case Q5LinearAddScheduleId::SlicedR32T32W2S2:
        return "linear_add.q5.sliced.r32.t32.w2.s2.residual";
    case Q5LinearAddScheduleId::SlicedR32T64W2S1:
        return "linear_add.q5.sliced.r32.t64.w2.s1.residual";
    case Q5LinearAddScheduleId::MmaResidualR32T32K128:
        return "linear_add.q5.mma.r32.t32.k128.residual";
    case Q5LinearAddScheduleId::MmaResidualR32T128:
        return "linear_add.q5.mma.r32.t128.residual";
    case Q5LinearAddScheduleId::MmaResidualR64T128:
        return "linear_add.q5.mma.r64.t128.cta_collective_residual";
    case Q5LinearAddScheduleId::MmaResidualR64T128Tail:
        return "linear_add.q5.mma.r64.t128.cta_collective_residual.narrow_tail";
    }
    return "linear_add.q5.unknown";
}

bool q5_linear_add_admits(const Q5LinearAddProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q5LinearAddPlan q5_linear_add_resolve_plan(const Q5LinearAddProblem& problem) {
    if (!q5_linear_add_admits(problem)) {
        throw std::invalid_argument("q5 linear_add: exact problem or column count is not admitted");
    }

    const auto resolve_from = [&](const auto& routes) -> Q5LinearAddPlan {
        for (const RouteSpec& route : routes) {
            if (route.cols.contains(problem.cols)) { return {route.schedule, 0}; }
        }
        throw std::logic_error("q5 linear_add: admitted problem has no covering route");
    };
    return problem.k == 6144 ? resolve_from(kK6144Routes) : resolve_from(kK17408Routes);
}

std::size_t q5_linear_add_capacity_workspace_bytes(std::int32_t rows, std::int32_t k,
                                                   std::int32_t padded_k, std::int32_t min_cols,
                                                   std::int32_t max_cols) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("q5 linear_add: invalid column interval");
    }
    (void)q5_linear_add_resolve_plan({rows, k, padded_k, min_cols});
    (void)q5_linear_add_resolve_plan({rows, k, padded_k, max_cols});

    return 0;
}

void q5_linear_add_execute_plan(const Q5LinearAddPlan& plan, const Tensor& x, const Weight& w,
                                Tensor& residual_out, WorkspaceArena& ws, cudaStream_t stream) {
    const Q5LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q5LinearAddPlan resolved = q5_linear_add_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("q5 linear_add: plan does not match the exact problem");
    }
    (void)ws;

    switch (plan.schedule) {
    case Q5LinearAddScheduleId::Split2ExactResidual:
        q5_linear_add_split2_exact_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SlicedR16T8W4S2:
        q5_linear_add_sliced_r16_t8_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SlicedR16T16W4S2:
        q5_linear_add_sliced_r16_t16_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SlicedR16T24W4S2:
        q5_linear_add_sliced_r16_t24_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SlicedR32T32W4S2:
        q5_linear_add_sliced_r32_t32_w4_s2_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SlicedR32T24W4S2Pairwise:
        q5_linear_add_sliced_r32_t24_pairwise_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SlicedR32T32W4S1:
        q5_linear_add_sliced_r32_t32_w4_s1_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SlicedR32T32W2S2:
        q5_linear_add_sliced_r32_t32_w2_s2_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SlicedR32T64W2S1:
        q5_linear_add_sliced_r32_t64_w2_s1_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR32T32K128:
        q5_linear_add_mma_r32_t32_k128_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR32T128:
        q5_linear_add_mma_r32_t128_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR64T128:
        q5_linear_add_mma_r64_t128_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR64T128Tail:
        launch_wide_with_narrow_tail(x, w, residual_out, ws, stream);
        return;
    }
    throw std::logic_error("q5 linear_add: unknown schedule");
}

void q5_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            WorkspaceArena& ws, cudaStream_t stream) {
    const Q5LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q5LinearAddPlan plan = q5_linear_add_resolve_plan(problem);
    q5_linear_add_execute_plan(plan, x, w, residual_out, ws, stream);
}

} // namespace ninfer::ops::detail
