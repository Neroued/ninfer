#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"

#include "core/device.h"
#include "ops/common/token_slices.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

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

constexpr std::array<RouteSpec, 2> kRoutes{{
    {{1, 16}, Q4Q5GdnInputScheduleId::IndependentDirectFixed},
    {{17, kAnyCols}, Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128},
}};

constexpr bool catalog_is_closed() noexcept {
    return kRoutes[0].cols.first == 1 && kRoutes[0].cols.last + 1 == kRoutes[1].cols.first &&
           kRoutes[1].cols.last == kAnyCols;
}

static_assert(catalog_is_closed(), "GDN input routes must be exact and closed");

bool supported_shape(const Q4Q5GdnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.qk_rows == 4096 && problem.value_rows == 6144 &&
           problem.output_rows == 10240 && problem.padded_k == 5120;
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
    case Q4Q5GdnInputScheduleId::IndependentDirectFixed:
        return "gdn_input_proj.q4_q5.independent_direct_fixed";
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128:
        return "gdn_input_proj.q4_q5.grouped_mixed.mma.r64.c128";
    }
    return "gdn_input_proj.q4_q5.unknown";
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
        Q4Q5GdnInputPlan plan{
            route.schedule,
            Q4KernelVariant::None,
            std::nullopt,
            0,
        };
        if (route.schedule == Q4Q5GdnInputScheduleId::IndependentDirectFixed) {
            plan.independent = Q4Q5GdnInputSubplans{
                q4_rowsplit_resolve_plan(
                    {problem.qk_rows, problem.input_rows, problem.padded_k, problem.cols}),
                q5_rowsplit_resolve_plan(
                    {problem.value_rows, problem.input_rows, problem.padded_k, problem.cols}),
            };
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

    return 0;
}

void q4_q5_gdn_input_execute_plan(const Q4Q5GdnInputPlan& plan, const Tensor& x,
                                  const Weight& qk_weight, const Weight& v_weight, Tensor& qkv,
                                  WorkspaceArena& ws, cudaStream_t stream) {
    const Q4Q5GdnInputProblem problem{
        x.ne[0], qk_weight.n, v_weight.n, qkv.ne[0], qk_weight.padded_shape[1], x.ne[1]};
    const Q4Q5GdnInputPlan resolved = q4_q5_gdn_input_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.grouped_variant != plan.grouped_variant ||
        !same_subplans(resolved.independent, plan.independent) ||
        resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("Q4/Q5 GDN input: plan does not match exact problem");
    }

    switch (plan.schedule) {
    case Q4Q5GdnInputScheduleId::IndependentDirectFixed: {
        (void)ws;
        Tensor qk    = qkv.slice(0, 0, problem.qk_rows);
        Tensor value = qkv.slice(0, problem.qk_rows, problem.value_rows);
        q4_rowsplit_launch_fixed_pitched(plan.independent->qk, x, qk_weight, qk, stream);
        q5_rowsplit_launch_fixed_pitched(plan.independent->value, x, v_weight, value, stream);
        return;
    }
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128:
        for_each_token_slice(problem.cols, 128, [&](std::int32_t offset, std::int32_t count) {
            const Tensor x_slice = x.slice(1, offset, count);
            Tensor qkv_slice     = qkv.slice(1, offset, count);
            q4_q5_gdn_input_grouped_mma_launch(plan.grouped_variant, x_slice, qk_weight, v_weight,
                                               qkv_slice, stream);
        });
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
