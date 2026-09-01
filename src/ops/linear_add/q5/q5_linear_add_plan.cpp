#include "ops/linear_add/q5/q5_linear_add_plan.h"

#include "ops/linear_add/q5/q5_linear_add_kernels.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/q5/q5_launch.h"
#include "ops/linear_add/q5/q5_linear_add_cutlass_sm70.h"
#endif

#include <algorithm>
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

#ifdef NINFER_VOLTA_BUILD
// MmaResidualR64C* need Ampere+ mma/ldmatrix and are trap-stubbed on sm_70.
// SimtWideTResidual (q5_linear_add_simt_wide_t_launch) is q5_rowsplit_gemm_simt_kernel with
// AddResidual=true and cols as a runtime grid parameter, so it covers every width above
// Split2ExactResidual's compile-time-switch ceiling -- but it's still plain SIMT, and this op
// (MLP down-projection at k=17408, attention gate/value at k=6144) was the single largest
// prefill kernel end to end (measured ~47% of total prefill GPU time at k=17408). Above the
// measured small-T routes, CutlassSm70TensorCoreResidual instead dequantizes
// to FP16 and runs NVIDIA CUTLASS's Sm70 (real mma.sync.m8n8k4) GEMM with beta=1 so the
// epilogue reads the existing residual as C and writes the sum back as D in place --
// independently verified (correctness against this exact SIMT kernel, L2 relative error
// ~0.17%; 19.6x measured speedup at the real N=5120/K=17408/T=256 down-proj shape on this
// V100). Cold-cache sweeps place both CUTLASS crossovers at T=17: for k=6144 the last SIMT
// win is T=16 (429us versus 439us), while for k=17408 the exact split2 route owns through T=16
// and the former wide-T SIMT route is already slower than CUTLASS at T=17. See docs/v100.md.
constexpr std::array<RouteSpec, 4> kK6144Routes{{
    {{1, 1}, Q5LinearAddScheduleId::GemvResidual},
    {{2, 13}, Q5LinearAddScheduleId::Split2ExactResidual},
    {{14, 16}, Q5LinearAddScheduleId::SimtWideTResidual},
    {{17, kAnyCols}, Q5LinearAddScheduleId::CutlassSm70TensorCoreResidual},
}};

constexpr std::array<RouteSpec, 3> kK17408Routes{{
    {{1, 1}, Q5LinearAddScheduleId::GemvResidual},
    {{2, 16}, Q5LinearAddScheduleId::Split2ExactResidual},
    {{17, kAnyCols}, Q5LinearAddScheduleId::CutlassSm70TensorCoreResidual},
}};
#else
constexpr std::array<RouteSpec, 6> kK6144Routes{{
    {{1, 1}, Q5LinearAddScheduleId::GemvResidual},
    {{2, 13}, Q5LinearAddScheduleId::Split2ExactResidual},
    {{14, 32}, Q5LinearAddScheduleId::MmaResidualR64C16},
    {{33, 48}, Q5LinearAddScheduleId::MmaResidualR64C24},
    {{49, 128}, Q5LinearAddScheduleId::MmaResidualR64C64},
    {{129, kAnyCols}, Q5LinearAddScheduleId::MmaResidualR64C128},
}};

constexpr std::array<RouteSpec, 6> kK17408Routes{{
    {{1, 1}, Q5LinearAddScheduleId::GemvResidual},
    {{2, 16}, Q5LinearAddScheduleId::Split2ExactResidual},
    {{17, 32}, Q5LinearAddScheduleId::MmaResidualR64C16},
    {{33, 48}, Q5LinearAddScheduleId::MmaResidualR64C24},
    {{49, 128}, Q5LinearAddScheduleId::MmaResidualR64C64},
    {{129, kAnyCols}, Q5LinearAddScheduleId::MmaResidualR64C128},
}};
#endif

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

} // namespace

const char* q5_linear_add_schedule_name(Q5LinearAddScheduleId schedule) noexcept {
    switch (schedule) {
    case Q5LinearAddScheduleId::GemvResidual:
        return "linear_add.q5.gemv.residual";
    case Q5LinearAddScheduleId::Split2ExactResidual:
        return "linear_add.q5.simt.split2.exact.residual";
    case Q5LinearAddScheduleId::MmaResidualR64C16:
        return "linear_add.q5.mma.r64.c16.cta_collective_residual";
    case Q5LinearAddScheduleId::MmaResidualR64C24:
        return "linear_add.q5.mma.r64.c24.cta_collective_residual";
    case Q5LinearAddScheduleId::MmaResidualR64C64:
        return "linear_add.q5.mma.r64.c64.cta_collective_residual";
    case Q5LinearAddScheduleId::MmaResidualR64C128:
        return "linear_add.q5.mma.r64.c128.cta_collective_residual";
    case Q5LinearAddScheduleId::SimtWideTResidual:
        return "linear_add.q5.simt.wide_t.residual";
    case Q5LinearAddScheduleId::CutlassSm70TensorCoreResidual:
        return "linear_add.q5.cutlass_sm70.tensor_core.residual";
    case Q5LinearAddScheduleId::VoltaMmaFusedResidual:
        return "linear_add.q5.volta_mma.fused.residual";
    }
    return "linear_add.q5.unknown";
}

#ifdef NINFER_VOLTA_BUILD
// Band for the fused tensor-core route. Lower edge is the measured crossover against whatever the
// table would otherwise pick, both routes through ninfer_q5_linear_add_bench back to back:
//   k=17408  T=9 353 vs 419us (table wins), T=12 634 vs 433, T=16 949 vs 456, T=32 1253 vs 552
//   k= 6144  T=9 118 vs 144us (table wins), T=12 150 vs 148, T=16 406 vs 154, T=32  459 vs 184
// Upper edge bounds the fp32 split-K accumulator (n*T*4; 655 KB at n=5120 T=32) and spans the
// concurrency range, since MTP3 puts C8 at T=32 and C16 at T=64.
constexpr std::int32_t kVoltaMmaMaxCols = 64;

// The lower edge follows the displaced schedule, not a single number. For k=6144 the table's
// Split2ExactResidual owns cols 2..13 and is genuinely good there (T=12: 149.5us against the
// fused route's 158.7us, a 6% loss), while SimtWideTResidual owns 14..16 and is not (T=16:
// 405.5us against 164.9us). For k=17408 Split2ExactResidual stretches to 16 and degrades
// earlier, so the fused route already wins at 12 (633.9us -> 469.0us).
constexpr std::int32_t q5_volta_mma_min_cols(std::int32_t k) noexcept {
    return k >= 12288 ? 12 : 14;
}

bool q5_linear_add_uses_volta_mma(const Q5LinearAddProblem& problem) noexcept {
    return problem.cols >= q5_volta_mma_min_cols(problem.k) &&
           problem.cols <= kVoltaMmaMaxCols &&
           q5_volta_mma_supported(problem.rows, problem.k, problem.cols);
}
#endif

bool q5_linear_add_admits(const Q5LinearAddProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q5LinearAddPlan q5_linear_add_resolve_plan(const Q5LinearAddProblem& problem) {
    if (!q5_linear_add_admits(problem)) {
        throw std::invalid_argument("q5 linear_add: exact problem or column count is not admitted");
    }

    const auto resolve_from = [&](const auto& routes) -> Q5LinearAddPlan {
        for (const RouteSpec& route : routes) {
            if (!route.cols.contains(problem.cols)) { continue; }
            std::size_t workspace_bytes = 0;
#ifdef NINFER_VOLTA_BUILD
            if (route.schedule == Q5LinearAddScheduleId::CutlassSm70TensorCoreResidual) {
                workspace_bytes =
                    q5_linear_add_cutlass_workspace_bytes(problem.rows, problem.k, problem.cols);
            }
            // Inside its band the fused tensor-core route replaces whatever the table selected,
            // so it becomes the schedule and owns the workspace figure outright. Reporting
            // max(displaced, fused) would over-state it, and the exact-workspace test compares
            // the queried figure against execution's high-water mark.
            if (q5_linear_add_uses_volta_mma(problem)) {
                return {Q5LinearAddScheduleId::VoltaMmaFusedResidual,
                        q5_volta_mma_workspace_bytes(problem.rows, problem.k, problem.cols)};
            }
#endif
            return {route.schedule, workspace_bytes};
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
    // CutlassSm70TensorCoreResidual's workspace is monotonic in cols (the FP16 weight-dequant
    // buffer is fixed at rows*k, the activation-cast buffer scales with cols), and the fused
    // tensor-core route's accumulator is rows*cols*4, also monotonic, so the true maximum over
    // [min_cols,max_cols] is always at one of the two endpoints.
    const Q5LinearAddPlan at_min = q5_linear_add_resolve_plan({rows, k, padded_k, min_cols});
    const Q5LinearAddPlan at_max = q5_linear_add_resolve_plan({rows, k, padded_k, max_cols});

    return std::max(at_min.workspace_bytes, at_max.workspace_bytes);
}

void q5_linear_add_execute_plan(const Q5LinearAddPlan& plan, const Tensor& x, const Weight& w,
                                Tensor& residual_out, WorkspaceArena& ws, cudaStream_t stream) {

    const Q5LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q5LinearAddPlan resolved = q5_linear_add_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("q5 linear_add: plan does not match the exact problem");
    }
#ifndef NINFER_VOLTA_BUILD
    (void)ws; // only CutlassSm70TensorCoreResidual (Volta-only) uses the workspace arena
#endif

    switch (plan.schedule) {
    case Q5LinearAddScheduleId::GemvResidual:
        q5_linear_add_gemv_residual_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::Split2ExactResidual:
        q5_linear_add_split2_exact_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR64C16:
        q5_linear_add_mma_r64_c16_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR64C24:
        q5_linear_add_mma_r64_c24_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR64C64:
        q5_linear_add_mma_r64_c64_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR64C128:
        q5_linear_add_mma_r64_c128_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SimtWideTResidual:
        q5_linear_add_simt_wide_t_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::VoltaMmaFusedResidual:
#ifdef NINFER_VOLTA_BUILD
        // Reads the residual as C and writes the sum back as D -- the same beta=1 epilogue
        // contract the CUTLASS schedule provides.
        launch_q5_volta_mma(x, w, residual_out, /*add_residual=*/true, /*weight_row_offset=*/0,
                            ws, stream);
        return;
#else
        throw std::logic_error("q5 linear_add: VoltaMmaFusedResidual is Volta-only");
#endif
    case Q5LinearAddScheduleId::CutlassSm70TensorCoreResidual:
#ifdef NINFER_VOLTA_BUILD
        q5_linear_add_cutlass_sm70_launch(x, w, residual_out, ws, stream);
        return;
#else
        throw std::logic_error("q5 linear_add: CutlassSm70TensorCoreResidual is Volta-only");
#endif
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
