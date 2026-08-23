#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"

#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_cutlass_sm70.h"
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/q5/q5_launch.h"
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
    Q4Q5GdnInputScheduleId schedule;
};

#ifdef NINFER_VOLTA_BUILD
// GroupedMixedMmaR64C128 needs Ampere+ mma/ldmatrix and is trap-stubbed on sm_70.
// IndependentDirectFixed's underlying kernels (q4_rowsplit_gemm_simt_kernel,
// q5_rowsplit_gemm_simt_kernel) are plain SIMT with cols as a runtime grid parameter, so it
// generalizes past T=16 unchanged and stays the route for small T -- decode and small prefill
// chunks, bandwidth-bound territory where CUTLASS's fixed (T-independent) dequant cost
// wouldn't pay for itself.
//
// For wide T (real prefill), CutlassSm70TensorCore dequantizes both parent weights once each
// (qk_weight is q4, value_z_weight is q5) to scratch FP16 buffers and runs three NVIDIA
// CUTLASS Sm70 (mma.sync.m8n8k4) GEMMs: qk (into qkv[0:4096]), value (into qkv[4096:10240],
// a pointer offset into the same dequant buffer as z, not a separate dequant pass), and z.
// This is the single largest remaining prefill cost this port found by profiling -- GDN layers
// vastly outnumber full-attention layers in this hybrid model, so this op's SIMT fallback
// dominated total prefill GPU time even after the other three CUTLASS conversions landed.
// Crossover measured directly (cold cache, 48-layer op, both routes forced across T): SIMT
// 781/1154/1400/2068us at T=16/24/32/48 against CUTLASS 1277/1276/1284/1290us. CUTLASS is
// essentially T-independent here -- its cost is the two fixed dequant passes (q4 qk_weight,
// q5 value_z_weight) -- so it wins from T=28 up, not T=64. The old 64 was inherited as a
// blanket port-wide default rather than measured for this op, and cost up to 1.6x in T=33..63.
// Band for the fused tensor-core route. This op is the single largest remaining consumer of
// memory traffic in a concurrent round: at T=32 the CUTLASS schedule costs 1284us and runs on 48
// of the 64 layers, ~32% of a C8 round, because it materialises both dequantised parents into
// global FP16 every call (52 MB read, 168 MB written, 168 MB read back). The fused kernels read
// each parent once at Q4/Q5 density.
constexpr std::int32_t kVoltaMmaMinCols = 12;
constexpr std::int32_t kVoltaMmaMaxCols = 64;

[[nodiscard]] inline bool gdn_uses_volta_mma(const Q4Q5GdnInputProblem& p) noexcept {
    return p.cols >= kVoltaMmaMinCols && p.cols <= kVoltaMmaMaxCols &&
           q4_volta_mma_supported(p.qk_rows, p.input_rows, p.cols) &&
           q5_volta_mma_supported(p.z_rows, p.input_rows, p.cols);
}

[[nodiscard]] inline std::size_t gdn_volta_mma_workspace_bytes(
    const Q4Q5GdnInputProblem& p) noexcept {
    // qk, value and z launch sequentially, each scoping its own fp32 partial, so the peak is the
    // largest of the three rather than their sum.
    const std::size_t a = q4_volta_mma_workspace_bytes(p.qk_rows, p.input_rows, p.cols);
    const std::size_t b = q5_volta_mma_workspace_bytes(p.z_rows, p.input_rows, p.cols);
    return a > b ? a : b;
}

constexpr std::array<RouteSpec, 2> kRoutes{{
    {{1, 27}, Q4Q5GdnInputScheduleId::IndependentDirectFixed},
    {{28, kAnyCols}, Q4Q5GdnInputScheduleId::CutlassSm70TensorCore},
}};

constexpr bool catalog_is_closed() noexcept {
    return kRoutes[0].cols.first == 1 && kRoutes[0].cols.last + 1 == kRoutes[1].cols.first &&
           kRoutes[1].cols.last == kAnyCols;
}
#else
constexpr std::array<RouteSpec, 2> kRoutes{{
    {{1, 16}, Q4Q5GdnInputScheduleId::IndependentDirectFixed},
    {{17, kAnyCols}, Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128},
}};

constexpr bool catalog_is_closed() noexcept {
    return kRoutes[0].cols.first == 1 && kRoutes[0].cols.last + 1 == kRoutes[1].cols.first &&
           kRoutes[1].cols.last == kAnyCols;
}
#endif

static_assert(catalog_is_closed(), "GDN input routes must be exact and closed");

bool supported_shape(const Q4Q5GdnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.qk_rows == 4096 && problem.value_z_rows == 12288 &&
           problem.qkv_rows == 10240 && problem.z_rows == 6144 && problem.padded_k == 5120;
}

} // namespace

const char* q4_q5_gdn_input_schedule_name(Q4Q5GdnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5GdnInputScheduleId::IndependentDirectFixed:
        return "gdn_input_proj.q4_q5.independent_direct_fixed";
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128:
        return "gdn_input_proj.q4_q5.grouped_mixed.mma.r64.c128";
    case Q4Q5GdnInputScheduleId::CutlassSm70TensorCore:
        return "gdn_input_proj.q4_q5.cutlass_sm70.tensor_core";
    case Q4Q5GdnInputScheduleId::VoltaMmaFused:
        return "gdn_input_proj.q4_q5.volta_mma.fused";
    }
    return "gdn_input_proj.q4_q5.unknown";
}

const char* q4_q5_gdn_input_conv_schedule_name(Q4Q5GdnInputConvScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5GdnInputConvScheduleId::ProjectionEpilogueFused:
        return "gdn_input_proj_conv.q4_q5.projection_epilogue_fused";
    case Q4Q5GdnInputConvScheduleId::Materialized:
        return "gdn_input_proj_conv.q4_q5.materialized";
    }
    return "gdn_input_proj_conv.q4_q5.unknown";
}

bool q4_q5_gdn_input_admits(const Q4Q5GdnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q4Q5GdnInputPlan q4_q5_gdn_input_resolve_plan(const Q4Q5GdnInputProblem& problem) {
    if (!q4_q5_gdn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 GDN input: exact problem or column count is not admitted");
    }

    for (const RouteSpec& route : kRoutes) {
        if (!route.cols.contains(problem.cols)) { continue; }
        Q4Q5GdnInputPlan plan{route.schedule, 0};
        switch (route.schedule) {
        case Q4Q5GdnInputScheduleId::IndependentDirectFixed:
        case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128:
#ifdef NINFER_VOLTA_BUILD
            if (gdn_uses_volta_mma(problem)) {
                return {Q4Q5GdnInputScheduleId::VoltaMmaFused,
                        gdn_volta_mma_workspace_bytes(problem)};
            }
#endif
            return plan;
        case Q4Q5GdnInputScheduleId::VoltaMmaFused:
            plan.workspace_bytes = gdn_volta_mma_workspace_bytes(problem);
            return plan;
        case Q4Q5GdnInputScheduleId::CutlassSm70TensorCore:
#ifdef NINFER_VOLTA_BUILD
            if (gdn_uses_volta_mma(problem)) {
                return {Q4Q5GdnInputScheduleId::VoltaMmaFused,
                        gdn_volta_mma_workspace_bytes(problem)};
            }
            plan.workspace_bytes = q4_q5_gdn_input_cutlass_workspace_bytes(problem.cols);
            return plan;
#else
            throw std::logic_error("Q4/Q5 GDN input: CutlassSm70TensorCore is Volta-only");
#endif
        }
    }
    throw std::logic_error("Q4/Q5 GDN input: admitted problem has no covering route");
}

std::size_t q4_q5_gdn_input_capacity_workspace_bytes(std::int32_t min_cols,
                                                      std::int32_t max_cols) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("Q4/Q5 GDN input: invalid column interval");
    }
    constexpr std::int32_t kInputRows = 5120, kQkRows = 4096, kValueZRows = 12288,
                           kQkvRows = 10240, kZRows = 6144;
    (void)q4_q5_gdn_input_resolve_plan(
        {kInputRows, kQkRows, kValueZRows, kQkvRows, kZRows, kInputRows, min_cols});
    (void)q4_q5_gdn_input_resolve_plan(
        {kInputRows, kQkRows, kValueZRows, kQkvRows, kZRows, kInputRows, max_cols});

    std::size_t maximum = 0;
    for (const RouteSpec& route : kRoutes) {
        if (route.cols.last < min_cols || route.cols.first > max_cols) { continue; }
        const std::int32_t endpoint = std::min(route.cols.last, max_cols);
        maximum = std::max(maximum,
                           q4_q5_gdn_input_resolve_plan({kInputRows, kQkRows, kValueZRows,
                                                         kQkvRows, kZRows, kInputRows, endpoint})
                               .workspace_bytes);
    }
    return maximum;
}

Q4Q5GdnInputConvPlan q4_q5_gdn_input_conv_resolve_plan(const Q4Q5GdnInputProblem& problem,
                                                       std::int32_t batch_size) {
    if (!q4_q5_gdn_input_admits(problem) || batch_size <= 0 || batch_size > 8) {
        throw std::invalid_argument(
            "Q4/Q5 GDN input conv: exact problem or column count is not admitted");
    }
    if (batch_size > 1) { return {Q4Q5GdnInputConvScheduleId::Materialized}; }
    switch (problem.cols) {
    case 1:
    case 2:
    case 3:
    case 5:
    case 6:
        return {Q4Q5GdnInputConvScheduleId::ProjectionEpilogueFused};
    default:
        return {Q4Q5GdnInputConvScheduleId::Materialized};
    }
}

void q4_q5_gdn_input_execute_plan(const Q4Q5GdnInputPlan& plan, const Tensor& x,
                                  const Weight& qk_weight, const Weight& value_z_weight,
                                  Tensor& qkv, Tensor& z, WorkspaceArena& workspace,
                                  cudaStream_t stream) {
    const Q4Q5GdnInputProblem problem{x.ne[0],   qk_weight.n, value_z_weight.n,
                                      qkv.ne[0], z.ne[0],     qk_weight.padded_shape[1],
                                      x.ne[1]};
    const Q4Q5GdnInputPlan resolved = q4_q5_gdn_input_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("Q4/Q5 GDN input: plan does not match exact problem");
    }

    switch (plan.schedule) {
    case Q4Q5GdnInputScheduleId::VoltaMmaFused: {
#ifdef NINFER_VOLTA_BUILD
        Tensor qk    = qkv.slice(0, 0, problem.qk_rows);
        Tensor value = qkv.slice(0, problem.qk_rows, problem.z_rows);
        // value_z_weight is one [2*z_rows, k] parent: rows [0,z_rows) feed value, the rest feed z.
        launch_q4_volta_mma(x, qk_weight, qk, workspace, stream);
        launch_q5_volta_mma(x, value_z_weight, value, /*add_residual=*/false,
                            /*weight_row_offset=*/0, workspace, stream);
        launch_q5_volta_mma(x, value_z_weight, z, /*add_residual=*/false,
                            /*weight_row_offset=*/problem.z_rows, workspace, stream);
        return;
#else
        throw std::logic_error("Q4/Q5 GDN input: VoltaMmaFused is Volta-only");
#endif
    }
    case Q4Q5GdnInputScheduleId::IndependentDirectFixed: {
        Tensor qk    = qkv.slice(0, 0, problem.qk_rows);
        Tensor value = qkv.slice(0, problem.qk_rows, problem.z_rows);
        q4_q5_gdn_input_independent_launch(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    }
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128:
        q4_q5_gdn_input_grouped_mma_launch(x, qk_weight, value_z_weight, qkv, z, stream);
        return;
    case Q4Q5GdnInputScheduleId::CutlassSm70TensorCore:
#ifdef NINFER_VOLTA_BUILD
        q4_q5_gdn_input_cutlass_sm70_launch(x, qk_weight, value_z_weight, qkv, z, workspace,
                                            stream);
        return;
#else
        throw std::logic_error("Q4/Q5 GDN input: CutlassSm70TensorCore is Volta-only");
#endif
    }
    throw std::logic_error("Q4/Q5 GDN input: unknown schedule");
}

void q4_q5_gdn_input_dispatch(const Tensor& x, const Weight& qk_weight,
                              const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                              WorkspaceArena& workspace, cudaStream_t stream) {
    const Q4Q5GdnInputProblem problem{x.ne[0],   qk_weight.n, value_z_weight.n,
                                      qkv.ne[0], z.ne[0],     qk_weight.padded_shape[1],
                                      x.ne[1]};
    const Q4Q5GdnInputPlan plan = q4_q5_gdn_input_resolve_plan(problem);
    q4_q5_gdn_input_execute_plan(plan, x, qk_weight, value_z_weight, qkv, z, workspace, stream);
}

} // namespace ninfer::ops::detail
