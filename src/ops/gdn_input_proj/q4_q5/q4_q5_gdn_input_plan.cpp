#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

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
    Q4Q5GdnInputScheduleId schedule;
};

constexpr std::array<RouteSpec, 2> kRoutes{{
    {{1, 16}, Q4Q5GdnInputScheduleId::MaterializedFixed},
    {{17, kQ4Q5GdnInputMaxCols}, Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128},
}};

constexpr bool catalog_is_closed() noexcept {
    return kRoutes[0].cols.first == 1 && kRoutes[0].cols.last + 1 == kRoutes[1].cols.first &&
           kRoutes[1].cols.last == kQ4Q5GdnInputMaxCols;
}

static_assert(catalog_is_closed(), "GDN input routes must be exact and closed");

bool supported_shape(const Q4Q5GdnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.qk_rows == 4096 && problem.value_rows == 6144 &&
           problem.output_rows == 10240 && problem.padded_k == 5120;
}

std::size_t checked_workspace_bytes(std::int32_t rows, std::int32_t cols) {
    const std::size_t r = static_cast<std::size_t>(rows);
    const std::size_t c = static_cast<std::size_t>(cols);
    if (c != 0 && r > std::numeric_limits<std::size_t>::max() / c) {
        throw std::overflow_error("Q4/Q5 GDN input: workspace element count overflows size_t");
    }
    const std::size_t elements = r * c;
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) {
        throw std::overflow_error("Q4/Q5 GDN input: workspace byte count overflows size_t");
    }
    return elements * sizeof(std::uint16_t);
}

bool same_q4_plan(Q4Plan lhs, Q4Plan rhs) {
    return lhs.schedule == rhs.schedule && lhs.variant == rhs.variant;
}

bool same_q5_plan(Q5Plan lhs, Q5Plan rhs) {
    return lhs.schedule == rhs.schedule && lhs.variant == rhs.variant;
}

bool same_subplans(const std::optional<Q4Q5GdnInputSubplans>& lhs,
                   const std::optional<Q4Q5GdnInputSubplans>& rhs) {
    if (lhs.has_value() != rhs.has_value()) { return false; }
    if (!lhs.has_value()) { return true; }
    return same_q4_plan(lhs->qk, rhs->qk) && same_q5_plan(lhs->value, rhs->value);
}

} // namespace

const char* q4_q5_gdn_input_schedule_name(Q4Q5GdnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5GdnInputScheduleId::MaterializedFixed:
        return "gdn_input_proj.q4_q5.materialized_fixed";
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128:
        return "gdn_input_proj.q4_q5.grouped_mixed.mma.r64.c128";
    }
    return "gdn_input_proj.q4_q5.unknown";
}

bool q4_q5_gdn_input_admits(const Q4Q5GdnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1 && problem.cols <= kQ4Q5GdnInputMaxCols;
}

Q4Q5GdnInputPlan q4_q5_gdn_input_resolve_plan(const Q4Q5GdnInputProblem& problem) {
    if (!q4_q5_gdn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 GDN input: exact problem or column count is not admitted");
    }

    for (const RouteSpec& route : kRoutes) {
        if (!route.cols.contains(problem.cols)) { continue; }
        Q4Q5GdnInputPlan plan{
            route.schedule,
            Q4KernelVariant::None,
            std::nullopt,
            0,
            problem.cols <= kQ4Q5GdnInputQualifiedCols,
        };
        if (route.schedule == Q4Q5GdnInputScheduleId::MaterializedFixed) {
            plan.materialized = Q4Q5GdnInputSubplans{
                q4_rowsplit_resolve_plan(
                    {problem.qk_rows, problem.input_rows, problem.padded_k, problem.cols}),
                q5_rowsplit_resolve_plan(
                    {problem.value_rows, problem.input_rows, problem.padded_k, problem.cols}),
            };
            plan.workspace_bytes = checked_workspace_bytes(problem.output_rows, problem.cols);
        } else {
            plan.grouped_variant =
                (problem.cols % 128) == 0 ? Q4KernelVariant::Full : Q4KernelVariant::Predicated;
        }
        return plan;
    }
    throw std::logic_error("Q4/Q5 GDN input: admitted problem has no covering route");
}

std::size_t q4_q5_gdn_input_capacity_workspace_bytes(std::int32_t qk_rows, std::int32_t value_rows,
                                                     std::int32_t max_cols) {
    constexpr std::int32_t output_rows = 10240;
    (void)q4_q5_gdn_input_resolve_plan({5120, qk_rows, value_rows, output_rows, 5120, max_cols});

    std::size_t maximum = 0;
    for (const RouteSpec& route : kRoutes) {
        if (route.cols.first > max_cols) { continue; }
        const std::int32_t endpoint = std::min(route.cols.last, max_cols);
        maximum                     = std::max(maximum, q4_q5_gdn_input_resolve_plan(
                                        {5120, qk_rows, value_rows, output_rows, 5120, endpoint})
                                                            .workspace_bytes);
    }
    return maximum;
}

void q4_q5_gdn_input_execute_plan(const Q4Q5GdnInputPlan& plan, const Tensor& x,
                                  const Weight& qk_weight, const Weight& v_weight, Tensor& qkv,
                                  WorkspaceArena& ws, cudaStream_t stream) {
    const Q4Q5GdnInputProblem problem{
        x.ne[0], qk_weight.n, v_weight.n, qkv.ne[0], qk_weight.padded_shape[1], x.ne[1]};
    const Q4Q5GdnInputPlan resolved = q4_q5_gdn_input_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.grouped_variant != plan.grouped_variant ||
        !same_subplans(resolved.materialized, plan.materialized) ||
        resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("Q4/Q5 GDN input: plan does not match exact problem");
    }

    switch (plan.schedule) {
    case Q4Q5GdnInputScheduleId::MaterializedFixed: {
        auto scratch_scope = ws.scope();
        Tensor qk          = ws.alloc(DType::BF16, {problem.qk_rows, problem.cols});
        Tensor value       = ws.alloc(DType::BF16, {problem.value_rows, problem.cols});
        q4_rowsplit_execute_plan(plan.materialized->qk, x, qk_weight, qk, stream);
        q5_rowsplit_execute_plan(plan.materialized->value, x, v_weight, value, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(
            qkv.data, static_cast<std::size_t>(problem.output_rows) * sizeof(std::uint16_t),
            qk.data, static_cast<std::size_t>(problem.qk_rows) * sizeof(std::uint16_t),
            static_cast<std::size_t>(problem.qk_rows) * sizeof(std::uint16_t), problem.cols,
            cudaMemcpyDeviceToDevice, stream));
        auto* value_dst = static_cast<unsigned char*>(qkv.data) +
                          static_cast<std::size_t>(problem.qk_rows) * sizeof(std::uint16_t);
        CUDA_CHECK(cudaMemcpy2DAsync(
            value_dst, static_cast<std::size_t>(problem.output_rows) * sizeof(std::uint16_t),
            value.data, static_cast<std::size_t>(problem.value_rows) * sizeof(std::uint16_t),
            static_cast<std::size_t>(problem.value_rows) * sizeof(std::uint16_t), problem.cols,
            cudaMemcpyDeviceToDevice, stream));
        return;
    }
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128:
        q4_q5_gdn_input_grouped_mma_launch(plan.grouped_variant, x, qk_weight, v_weight, qkv,
                                           stream);
        return;
    }
    throw std::logic_error("Q4/Q5 GDN input: unknown schedule");
}

void q4_q5_gdn_input_dispatch(const Tensor& x, const Weight& qk_weight, const Weight& v_weight,
                              Tensor& qkv, WorkspaceArena& ws, cudaStream_t stream) {
    const Q4Q5GdnInputProblem problem{
        x.ne[0], qk_weight.n, v_weight.n, qkv.ne[0], qk_weight.padded_shape[1], x.ne[1]};
    const Q4Q5GdnInputPlan plan = q4_q5_gdn_input_resolve_plan(problem);
    q4_q5_gdn_input_execute_plan(plan, x, qk_weight, v_weight, qkv, ws, stream);
}

} // namespace ninfer::ops::detail
