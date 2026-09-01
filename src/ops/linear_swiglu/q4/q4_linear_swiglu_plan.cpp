#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"

#include "ninfer/ops/linear.h"
#include "ninfer/ops/silu_mul.h"
#include "core/layout.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"
#include "ops/linear/q4/q4_launch.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_cutlass_sm70.h"
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

struct RouteSpec {
    ColsSet cols;
    Q4LinearSwiGluScheduleId schedule;
};

constexpr Q4LinearSwiGluProblem kShape{34816, 17408, 5120, 5120, 1};

#ifdef NINFER_VOLTA_BUILD
// Lower edge of the fused tensor-core route. At exactly 16 it is a wash (that is q4_simt_r8_c8's
// best point -- two exact eight-column tiles); it wins from 17 up, and SmallTExact's own ceiling
// of 32 bounds the split-K buffer.
constexpr std::int32_t kVoltaMmaMinCols = 16;
#endif

#ifdef NINFER_VOLTA_BUILD
// MmaSplitHalfPairR32C*/Materialized-via-linear() route through Ampere+ mma/ldmatrix kernels
// (q4_rowsplit_gemm_mma.cuh and friends), trap-stubbed on sm_70. SmallTExact's Volta compose
// (a tuned q4 SIMT launch + silu_mul) is fully general over T via for_each_token_slice, and stays
// the route for small T -- decode and small prefill chunks, where this GEMM is bandwidth-bound
// and CUTLASS's fixed (T-independent) dequant cost wouldn't pay for itself.
//
// For wide T (real prefill), CutlassSm70TensorCore instead dequantizes to a scratch FP16
// buffer and runs NVIDIA CUTLASS's Sm70 (real mma.sync.m8n8k4 tensor-core) GEMM template --
// independently verified (correctness against this exact SIMT kernel, L2 relative error
// ~0.17%, well within expected fp16-storage precision differences; 6.6x measured speedup at
// the real N=34816/K=5120/T=256 gate_up shape on this V100). Cold-cache measurement places
// the exact crossover after T=32: the C8 SIMT route re-reads the weights for a fifth tile at
// T=33, while CUTLASS keeps one dequant pass and one partially occupied 128-row M tile.
constexpr std::array<RouteSpec, 3> kRoutes{{
    {{1, 1}, Q4LinearSwiGluScheduleId::GemvPair},
    {{2, 32}, Q4LinearSwiGluScheduleId::SmallTExact},
    {{33, kAnyCols}, Q4LinearSwiGluScheduleId::CutlassSm70TensorCore},
}};
#else
constexpr std::array<RouteSpec, 10> kRoutes{{
    {{1, 1}, Q4LinearSwiGluScheduleId::GemvPair},
    {{2, 32}, Q4LinearSwiGluScheduleId::SmallTExact},
    {{33, 40}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C40},
    {{41, 48}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C48},
    {{49, 128}, Q4LinearSwiGluScheduleId::Materialized},
    {{129, 256}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128},
    {{257, 384}, Q4LinearSwiGluScheduleId::Materialized},
    {{385, 512}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128},
    {{513, 640}, Q4LinearSwiGluScheduleId::Materialized},
    {{641, kAnyCols}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128},
}};
#endif

constexpr bool catalog_is_closed() noexcept {
    std::int64_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.cols.first != expected || route.cols.last < route.cols.first) { return false; }
        expected = static_cast<std::int64_t>(route.cols.last) + 1;
    }
    return kRoutes.back().cols.last == kAnyCols &&
           expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(), "Q4 LinearSwiGLU routes must be exact, contiguous, and closed");

bool supported_shape(const Q4LinearSwiGluProblem& problem) noexcept {
    return problem.gate_up_rows == kShape.gate_up_rows &&
           problem.output_rows == kShape.output_rows && problem.k == kShape.k &&
           problem.padded_k == kShape.padded_k;
}

template <class Allocator>
Tensor allocate_materialized_workspace(Allocator& allocator, std::int32_t rows, std::int32_t cols) {
    return allocator.alloc(DType::BF16, {rows, cols});
}

std::size_t materialized_workspace_bytes(std::int32_t rows, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_materialized_workspace(layout, rows, cols);
    return layout.peak_bytes(1);
}

} // namespace

const char* q4_linear_swiglu_schedule_name(Q4LinearSwiGluScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4LinearSwiGluScheduleId::GemvPair:
        return "linear_swiglu.q4.gemv.paired_rows";
    case Q4LinearSwiGluScheduleId::SmallTExact:
        return "linear_swiglu.q4.mma.small_t.exact";
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C40:
        return "linear_swiglu.q4.mma.split_half_pair.r32.c40";
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C48:
        return "linear_swiglu.q4.mma.split_half_pair.r32.c48";
    case Q4LinearSwiGluScheduleId::Materialized:
        return "linear_swiglu.q4.materialized";
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128:
        return "linear_swiglu.q4.mma.split_half_pair.r32.c128";
    case Q4LinearSwiGluScheduleId::CutlassSm70TensorCore:
        return "linear_swiglu.q4.cutlass_sm70.tensor_core";
    }
    return "linear_swiglu.q4.unknown";
}

bool q4_linear_swiglu_admits(const Q4LinearSwiGluProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
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
            0,
        };
        switch (route.schedule) {
        case Q4LinearSwiGluScheduleId::SmallTExact:
#ifdef NINFER_VOLTA_BUILD
            // q4_small_t_mma_kernel (the fused tensor-core path this schedule normally
            // uses) traps below sm_80 — see q4_small_t_mma.cuh. On Volta this schedule
            // composes from the same already-proven pieces Materialized below uses
            // (general linear() into an unfused workspace, then silu_mul), so it needs
            // the same real workspace_bytes accounting Materialized does, not 0.
            plan.workspace_bytes = materialized_workspace_bytes(problem.gate_up_rows, problem.cols);
            if (problem.cols >= kVoltaMmaMinCols) {
                // The fused tensor-core gate_up route additionally needs an fp32 split-K
                // accumulator (gate_up_rows * cols * 4 bytes).
                plan.workspace_bytes +=
                    q4_volta_mma_workspace_bytes(problem.gate_up_rows, problem.k, problem.cols);
            }
            return plan;
#endif
            return plan;
        case Q4LinearSwiGluScheduleId::GemvPair:
        case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C40:
        case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C48:
            return plan;
        case Q4LinearSwiGluScheduleId::Materialized:
            plan.workspace_bytes = materialized_workspace_bytes(problem.gate_up_rows, problem.cols);
            return plan;
        case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128:
            return plan;
        case Q4LinearSwiGluScheduleId::CutlassSm70TensorCore:
#ifdef NINFER_VOLTA_BUILD
            // gate_up output buffer (materialized_workspace_bytes) plus the CUTLASS launcher's
            // own internal scratch (dequant FP16 weight buffer -- fixed size, independent of
            // cols -- FP16-cast activations, and CUTLASS's own GEMM workspace).
            plan.workspace_bytes =
                materialized_workspace_bytes(problem.gate_up_rows, problem.cols) +
                q4_linear_swiglu_cutlass_workspace_bytes(problem.gate_up_rows, problem.k,
                                                         problem.cols);
            return plan;
#else
            throw std::logic_error("q4 linear_swiglu: CutlassSm70TensorCore is Volta-only");
#endif
        }
    }
    throw std::logic_error("q4 linear_swiglu: admitted problem has no covering route");
}

std::size_t q4_linear_swiglu_capacity_workspace_bytes(std::int32_t gate_up_rows,
                                                      std::int32_t output_rows, std::int32_t k,
                                                      std::int32_t padded_k, std::int32_t min_cols,
                                                      std::int32_t max_cols) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("q4 linear_swiglu: invalid column interval");
    }
    (void)q4_linear_swiglu_resolve_plan({gate_up_rows, output_rows, k, padded_k, min_cols});
    (void)q4_linear_swiglu_resolve_plan({gate_up_rows, output_rows, k, padded_k, max_cols});

    std::size_t maximum = 0;
    for (const RouteSpec& route : kRoutes) {
        if (route.cols.last < min_cols || route.cols.first > max_cols) { continue; }
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
    if (resolved.schedule != plan.schedule || resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("q4 linear_swiglu: plan does not match the exact problem");
    }

    switch (plan.schedule) {
    case Q4LinearSwiGluScheduleId::GemvPair:
        q4_linear_swiglu_gemv_pair_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::SmallTExact:
#ifdef NINFER_VOLTA_BUILD
        {
            // Same compose as Materialized below, forcing the gate_up GEMM through the
            // known-Volta-safe SIMT kernel directly rather than the general linear()
            // dispatcher: linear() would pick launch_q4_mma_r64_c128 for the upper half
            // of this schedule's T range (17..32), which is a different, also-unported
            // tensor-core kernel. The four-column schedule is materially faster for the exact
            // T=2..4 MTP verify widths; the eight-column schedule owns the rest of this interval.
            auto scratch_scope = ws.scope();
            Tensor gate_up = allocate_materialized_workspace(ws, problem.gate_up_rows, problem.cols);
            if (q4_uses_volta_qpn(problem.gate_up_rows, problem.k, problem.cols)) {
                // Quadpair-split-N: below the fused 32x8 route's band, where that geometry would
                // pad most of its A rows away. On this exact shape it measures 309.2us at T=8
                // against sliced SIMT's 452.6, and 306.2 against 567.3 at T=7 -- see q4_launch.h.
                launch_q4_volta_qpn(x, w, gate_up, stream);
            } else if (problem.cols >= kVoltaMmaMinCols &&
                q4_volta_mma_supported(problem.gate_up_rows, problem.k, problem.cols)) {
                // Fused dequant + mma.sync.m8n8k4. Measured on the gate_up shape's sibling
                // [4096,5120] against q4_simt_r8_c8, back to back through the same bench:
                // T=16 141.3 -> 138.2us, T=24 199.7 -> 143.4us, T=32 259.1 -> 148.5us.
                launch_q4_volta_mma(x, w, gate_up, ws, stream);
            } else if (problem.cols <= 4) {
                launch_q4_simt_r8_c4(x, w, gate_up, stream);
            } else {
                launch_q4_simt_r8_c8(x, w, gate_up, stream);
            }
            silu_mul(gate_up.slice(0, 0, problem.output_rows),
                    gate_up.slice(0, problem.output_rows, problem.output_rows), out, stream);
            return;
        }
#endif
        q4_linear_swiglu_small_t_exact_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C40:
        q4_linear_swiglu_mma_split_half_pair_r32_c40_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C48:
        q4_linear_swiglu_mma_split_half_pair_r32_c48_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::Materialized: {
        auto scratch_scope = ws.scope();
        Tensor gate_up = allocate_materialized_workspace(ws, problem.gate_up_rows, problem.cols);
        linear(x, w, gate_up, stream);
        silu_mul(gate_up.slice(0, 0, problem.output_rows),
                 gate_up.slice(0, problem.output_rows, problem.output_rows), out, stream);
        return;
    }
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128:
        q4_linear_swiglu_mma_split_half_pair_r32_c128_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::CutlassSm70TensorCore:
#ifdef NINFER_VOLTA_BUILD
        {
            auto scratch_scope = ws.scope();
            Tensor gate_up = allocate_materialized_workspace(ws, problem.gate_up_rows, problem.cols);
            q4_linear_swiglu_cutlass_sm70_launch(x, w, gate_up, ws, stream);
            silu_mul(gate_up.slice(0, 0, problem.output_rows),
                    gate_up.slice(0, problem.output_rows, problem.output_rows), out, stream);
            return;
        }
#else
        throw std::logic_error("q4 linear_swiglu: CutlassSm70TensorCore is Volta-only");
#endif
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
