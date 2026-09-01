#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"

#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_cutlass_sm70.h"
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/q5/q5_launch.h"
#endif

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols     = std::numeric_limits<std::int32_t>::max();
constexpr std::int32_t kHiddenShape = 5120;
constexpr std::int32_t kQRows       = 6144;
constexpr std::int32_t kKvRows      = 1024;

struct ColsSet {
    std::int32_t first;
    std::int32_t last;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last;
    }
};

struct RouteSpec {
    ColsSet cols;
    Q4Q5AttnInputScheduleId schedule;
};

#ifdef NINFER_VOLTA_BUILD
// GroupedHomogeneousPairMma* need Ampere+ mma/ldmatrix and are trap-stubbed on sm_70.
// ParentSplitFixed's underlying kernels take cols as a runtime grid parameter (see
// q4_q5_attn_input_small_t.cu), so it generalizes past T=16 unchanged and stays the route for
// small T -- decode and small prefill chunks, bandwidth-bound territory where CUTLASS's fixed
// (T-independent) dequant cost wouldn't pay for itself.
//
// For wide T (real prefill), CutlassSm70TensorCore dequantizes both parent weights once each
// (query_key is q4, gate_value is q5) to scratch FP16 buffers and runs four NVIDIA CUTLASS
// Sm70 (mma.sync.m8n8k4) GEMMs -- one per output (q, k, gate, v), each a pointer offset into
// its shared dequant buffer, not a separate dequant pass. Crossover measured directly (cold cache,
// both routes forced): SIMT 522/763/1006/1490/1731us at T=16/24/32/48/56 against CUTLASS
// 1324/1326/1291/1295/~1297us, so the true crossover is ~40, not the blanket 64 this port
// applied. SIMT still wins at T<=32, so this does not affect decode concurrency (C8 with MTP3
// is T=32); it recovers up to 1.33x in the T=40..63 band. Contrast gdn_input_proj, whose larger
// SIMT work (16384 vs 7168 parent rows) against a similar fixed dequant cost puts its crossover
// at 28. See docs/v100.md.
// Band for the fused tensor-core route (ops/linear/q{4,5}/q{4,5}_volta_mma_gemm.cuh), the same
// change gdn_input_proj took. Both stored parents are [7168,5120] with rows 0..6143 feeding
// q/gate and 6144..7167 feeding k/v, and both fused launchers take a weight_row_offset, so this
// is four launches over the existing kernels rather than any new arithmetic. 16 of the 64 layers
// run this op.
//
// Neither incumbent suits this range. ParentSplitFixed is SIMT and re-reads its parent once per
// 8 output columns; CutlassSm70 materialises both dequantised parents into global FP16 on every
// call, so its cost is nearly T-independent.
//
// Both band edges are measured against the route they displace, cold cache, same session, the
// whole op (all four projections), microseconds on a V100-SXM2-32GB:
//
//   T:          7     8     9    10    12    16    24    32    48    64    80    96   112   128
//   incumbent 316   275   434   443   399   454   662   870  1269  1275  1285  1291  1304  1310
//   fused     311   307   312   314   317   339   364   389   603   634   888   924  1287  1336
//
// (incumbent is ParentSplitFixed through T=39 and CutlassSm70TensorCore from 40.)
//
// The lower edge is a cliff in the incumbent, not a slope in the fused route: the SIMT schedule
// tiles 8 output columns, so T=9 is where it starts paying for a second column tile it barely
// uses -- 275us at T=8 against 434us at T=9. The fused route is flat across the same span. Hence
// 9 exactly.
//
// The upper edge is the mirror image, in the fused route's own geometry: it re-reads both parents
// once per 32-column A tile, so cost steps at T=33/65/97. Three tiles still beats CUTLASS's fixed
// dequant comfortably (1.40x at T=96); four does not (1287 against 1304 at T=112, and losing from
// 128). So the band ends at the last tile boundary that wins outright.
constexpr std::int32_t kVoltaMmaMinCols = 9;
constexpr std::int32_t kVoltaMmaMaxCols = 96;

[[nodiscard]] inline bool attn_uses_volta_mma(const Q4Q5AttnInputProblem& p) noexcept {
    return p.cols >= kVoltaMmaMinCols && p.cols <= kVoltaMmaMaxCols &&
           q4_volta_mma_supported(p.query_rows, p.input_rows, p.cols) &&
           q4_volta_mma_supported(p.kv_rows, p.input_rows, p.cols) &&
           q5_volta_mma_supported(p.query_rows, p.input_rows, p.cols) &&
           q5_volta_mma_supported(p.kv_rows, p.input_rows, p.cols);
}

[[nodiscard]] inline std::size_t attn_volta_mma_workspace_bytes(
    const Q4Q5AttnInputProblem& p) noexcept {
    // The four launches are sequential and each scopes its own fp32 partial, so the peak is the
    // largest of them, not their sum. All four have to be asked: the figure is no longer monotonic
    // in n, because a shape that resolves to a single split needs no accumulator at all. From
    // T=33 the query-width launches are exactly that while the kv-width ones, with only 32 CTAs
    // to their name, still split K and still need the buffer.
    const std::size_t sizes[4] = {
        q4_volta_mma_workspace_bytes(p.query_rows, p.input_rows, p.cols),
        q4_volta_mma_workspace_bytes(p.kv_rows, p.input_rows, p.cols),
        q5_volta_mma_workspace_bytes(p.query_rows, p.input_rows, p.cols),
        q5_volta_mma_workspace_bytes(p.kv_rows, p.input_rows, p.cols),
    };
    return *std::max_element(std::begin(sizes), std::end(sizes));
}

constexpr std::array<RouteSpec, 2> kRoutes{{
    {{1, 39}, Q4Q5AttnInputScheduleId::ParentSplitFixed},
    {{40, kAnyCols}, Q4Q5AttnInputScheduleId::CutlassSm70TensorCore},
}};

constexpr bool catalog_is_closed() noexcept {
    return kRoutes[0].cols.first == 1 && kRoutes[0].cols.last + 1 == kRoutes[1].cols.first &&
           kRoutes[1].cols.last == kAnyCols;
}
#else
constexpr std::array<RouteSpec, 3> kRoutes{{
    {{1, 16}, Q4Q5AttnInputScheduleId::ParentSplitFixed},
    {{17, 20}, Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR16C64S3},
    {{21, kAnyCols}, Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR32C64S4},
}};

constexpr bool catalog_is_closed() noexcept {
    return kRoutes[0].cols.first == 1 && kRoutes[0].cols.last + 1 == kRoutes[1].cols.first &&
           kRoutes[1].cols.last + 1 == kRoutes[2].cols.first && kRoutes[2].cols.last == kAnyCols;
}
#endif

static_assert(catalog_is_closed(), "attention input routes must be exact and closed");

bool supported_shape(const Q4Q5AttnInputProblem& problem) noexcept {
    return problem.input_rows == kHiddenShape && problem.query_rows == kQRows &&
           problem.kv_rows == kKvRows && problem.padded_k == kHiddenShape;
}

} // namespace

const char* q4_q5_attn_input_schedule_name(Q4Q5AttnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        return "attn_input_proj.q4_q5.parent_split_fixed";
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR16C64S3:
        return "attn_input_proj.q4_q5.grouped_homogeneous_pair.mma.r16.c64.s3";
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR32C64S4:
        return "attn_input_proj.q4_q5.grouped_homogeneous_pair.mma.r32.c64.s4";
    case Q4Q5AttnInputScheduleId::CutlassSm70TensorCore:
        return "attn_input_proj.q4_q5.cutlass_sm70.tensor_core";
    case Q4Q5AttnInputScheduleId::VoltaMmaFused:
        return "attn_input_proj.q4_q5.volta_mma.fused";
    }
    return "attn_input_proj.q4_q5.unknown";
}

bool q4_q5_attn_input_admits(const Q4Q5AttnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q4Q5AttnInputPlan q4_q5_attn_input_resolve_plan(const Q4Q5AttnInputProblem& problem) {
    if (!q4_q5_attn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 attention input: exact problem or column count is not admitted");
    }

    for (const RouteSpec& route : kRoutes) {
        if (!route.cols.contains(problem.cols)) { continue; }
        Q4Q5AttnInputPlan plan{route.schedule, 0};
        switch (route.schedule) {
        case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR16C64S3:
        case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR32C64S4:
#ifdef NINFER_VOLTA_BUILD
            // Inside its band the fused route replaces whatever the table selected, so it becomes
            // the schedule and owns the workspace figure outright.
            if (attn_uses_volta_mma(problem)) {
                return {Q4Q5AttnInputScheduleId::VoltaMmaFused,
                        attn_volta_mma_workspace_bytes(problem)};
            }
#endif
            return plan;
        case Q4Q5AttnInputScheduleId::VoltaMmaFused:
#ifdef NINFER_VOLTA_BUILD
            plan.workspace_bytes = attn_volta_mma_workspace_bytes(problem);
            return plan;
#else
            throw std::logic_error("Q4/Q5 attention input: VoltaMmaFused is Volta-only");
#endif
        case Q4Q5AttnInputScheduleId::CutlassSm70TensorCore:
#ifdef NINFER_VOLTA_BUILD
            if (attn_uses_volta_mma(problem)) {
                return {Q4Q5AttnInputScheduleId::VoltaMmaFused,
                        attn_volta_mma_workspace_bytes(problem)};
            }
            plan.workspace_bytes = q4_q5_attn_input_cutlass_workspace_bytes(problem.cols);
            return plan;
#else
            throw std::logic_error("Q4/Q5 attention input: CutlassSm70TensorCore is Volta-only");
#endif
        }
    }
    throw std::logic_error("Q4/Q5 attention input: admitted problem has no covering route");
}

std::size_t q4_q5_attn_input_capacity_workspace_bytes(std::int32_t min_cols,
                                                       std::int32_t max_cols) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("Q4/Q5 attention input: invalid column interval");
    }
    (void)q4_q5_attn_input_resolve_plan({kHiddenShape, kQRows, kKvRows, kHiddenShape, min_cols});
    (void)q4_q5_attn_input_resolve_plan({kHiddenShape, kQRows, kKvRows, kHiddenShape, max_cols});

    const auto at = [](std::int32_t cols) {
        return q4_q5_attn_input_resolve_plan({kHiddenShape, kQRows, kKvRows, kHiddenShape, cols})
            .workspace_bytes;
    };

    std::size_t maximum = 0;
    for (const RouteSpec& route : kRoutes) {
        if (route.cols.last < min_cols || route.cols.first > max_cols) { continue; }
        maximum = std::max(maximum, at(std::min(route.cols.last, max_cols)));
    }
#ifdef NINFER_VOLTA_BUILD
    // The fused route displaces the table inside [kVoltaMmaMinCols, kVoltaMmaMaxCols], which cuts
    // across kRoutes' second span. Probing only the table's own endpoints therefore misses the
    // fused band's right edge whenever the interval runs past it into CutlassSm70TensorCore.
    if (min_cols <= kVoltaMmaMaxCols && max_cols >= kVoltaMmaMinCols) {
        maximum = std::max(maximum, at(std::min(kVoltaMmaMaxCols, max_cols)));
    }
#endif
    return maximum;
}

void q4_q5_attn_input_execute_plan(const Q4Q5AttnInputPlan& plan, const Tensor& x,
                                   const Weight& query_key_weight, const Weight& gate_value_weight,
                                   Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    const Q4Q5AttnInputPlan resolved = q4_q5_attn_input_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("Q4/Q5 attention input: plan does not match exact problem");
    }

    switch (plan.schedule) {
    case Q4Q5AttnInputScheduleId::VoltaMmaFused:
#ifdef NINFER_VOLTA_BUILD
        // Each stored parent is one [query_rows + kv_rows, k] weight: rows [0,query_rows) feed
        // q (resp. gate), the rest feed k (resp. v).
        launch_q4_volta_mma(x, query_key_weight, q, workspace, stream, /*weight_row_offset=*/0);
        launch_q4_volta_mma(x, query_key_weight, k, workspace, stream,
                            /*weight_row_offset=*/problem.query_rows);
        launch_q5_volta_mma(x, gate_value_weight, gate, /*add_residual=*/false,
                            /*weight_row_offset=*/0, workspace, stream);
        launch_q5_volta_mma(x, gate_value_weight, v, /*add_residual=*/false,
                            /*weight_row_offset=*/problem.query_rows, workspace, stream);
        return;
#else
        throw std::logic_error("Q4/Q5 attention input: VoltaMmaFused is Volta-only");
#endif
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        q4_q5_attn_input_small_t_launch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                        stream);
        return;
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR16C64S3:
        q4_q5_attn_input_grouped_mma_r16_c64_s3_launch(x, query_key_weight, gate_value_weight, q,
                                                       gate, k, v, stream);
        return;
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR32C64S4:
        q4_q5_attn_input_grouped_mma_r32_c64_s4_launch(x, query_key_weight, gate_value_weight, q,
                                                       gate, k, v, stream);
        return;
    case Q4Q5AttnInputScheduleId::CutlassSm70TensorCore:
#ifdef NINFER_VOLTA_BUILD
        q4_q5_attn_input_cutlass_sm70_launch(x, query_key_weight, gate_value_weight, q, gate, k,
                                             v, workspace, stream);
        return;
#else
        throw std::logic_error("Q4/Q5 attention input: CutlassSm70TensorCore is Volta-only");
#endif
    }
    throw std::logic_error("Q4/Q5 attention input: unknown schedule");
}

void q4_q5_attn_input_dispatch(const Tensor& x, const Weight& query_key_weight,
                               const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k,
                               Tensor& v, WorkspaceArena& workspace, cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    const Q4Q5AttnInputPlan plan = q4_q5_attn_input_resolve_plan(problem);
    q4_q5_attn_input_execute_plan(plan, x, query_key_weight, gate_value_weight, q, gate, k, v,
                                  workspace, stream);
}

} // namespace ninfer::ops::detail
