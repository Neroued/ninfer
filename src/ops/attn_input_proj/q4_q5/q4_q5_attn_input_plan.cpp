#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"

#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include <array>
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
    Q4Q5AttnInputScheduleId schedule;
};

constexpr std::array<RouteSpec, 2> kRoutes{{
    {{1, 16}, Q4Q5AttnInputScheduleId::IndependentFixed},
    {{17, kQ4Q5AttnInputMaxCols}, Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR64C128},
}};

constexpr bool catalog_is_closed() noexcept {
    return kRoutes[0].cols.first == 1 && kRoutes[0].cols.last + 1 == kRoutes[1].cols.first &&
           kRoutes[1].cols.last == kQ4Q5AttnInputMaxCols;
}

static_assert(catalog_is_closed(), "attention input routes must be exact and closed");

bool supported_shape(const Q4Q5AttnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.query_rows == 6144 && problem.kv_rows == 1024 &&
           problem.padded_k == 5120;
}

bool same_q4_plan(Q4Plan lhs, Q4Plan rhs) {
    return lhs.schedule == rhs.schedule && lhs.variant == rhs.variant;
}

bool same_q5_plan(Q5Plan lhs, Q5Plan rhs) {
    return lhs.schedule == rhs.schedule && lhs.variant == rhs.variant;
}

bool same_subplans(const std::optional<Q4Q5AttnInputSubplans>& lhs,
                   const std::optional<Q4Q5AttnInputSubplans>& rhs) {
    if (lhs.has_value() != rhs.has_value()) { return false; }
    if (!lhs.has_value()) { return true; }
    return same_q4_plan(lhs->query, rhs->query) && same_q5_plan(lhs->gate, rhs->gate) &&
           same_q4_plan(lhs->key, rhs->key) && same_q5_plan(lhs->value, rhs->value);
}

} // namespace

const char* q4_q5_attn_input_schedule_name(Q4Q5AttnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5AttnInputScheduleId::IndependentFixed:
        return "attn_input_proj.q4_q5.independent_fixed";
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR64C128:
        return "attn_input_proj.q4_q5.grouped_homogeneous_pair.mma.r64.c128";
    }
    return "attn_input_proj.q4_q5.unknown";
}

bool q4_q5_attn_input_admits(const Q4Q5AttnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1 && problem.cols <= kQ4Q5AttnInputMaxCols;
}

Q4Q5AttnInputPlan q4_q5_attn_input_resolve_plan(const Q4Q5AttnInputProblem& problem) {
    if (!q4_q5_attn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 attention input: exact problem or column count is not admitted");
    }

    for (const RouteSpec& route : kRoutes) {
        if (!route.cols.contains(problem.cols)) { continue; }
        Q4Q5AttnInputPlan plan{
            route.schedule,
            Q4KernelVariant::None,
            std::nullopt,
            0,
            problem.cols <= kQ4Q5AttnInputQualifiedCols,
        };
        if (route.schedule == Q4Q5AttnInputScheduleId::IndependentFixed) {
            plan.independent = Q4Q5AttnInputSubplans{
                q4_rowsplit_resolve_plan(
                    {problem.query_rows, problem.input_rows, problem.padded_k, problem.cols}),
                q5_rowsplit_resolve_plan(
                    {problem.query_rows, problem.input_rows, problem.padded_k, problem.cols}),
                q4_rowsplit_resolve_plan(
                    {problem.kv_rows, problem.input_rows, problem.padded_k, problem.cols}),
                q5_rowsplit_resolve_plan(
                    {problem.kv_rows, problem.input_rows, problem.padded_k, problem.cols}),
            };
        } else {
            plan.grouped_variant =
                (problem.cols % 128) == 0 ? Q4KernelVariant::Full : Q4KernelVariant::Predicated;
        }
        return plan;
    }
    throw std::logic_error("Q4/Q5 attention input: admitted problem has no covering route");
}

void q4_q5_attn_input_execute_plan(const Q4Q5AttnInputPlan& plan, const Tensor& x,
                                   const Weight& q_weight, const Weight& gate_weight,
                                   const Weight& k_weight, const Weight& v_weight, Tensor& q,
                                   Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], q_weight.padded_shape[1],
                                       x.ne[1]};
    const Q4Q5AttnInputPlan resolved = q4_q5_attn_input_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.grouped_variant != plan.grouped_variant ||
        !same_subplans(resolved.independent, plan.independent) ||
        resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("Q4/Q5 attention input: plan does not match exact problem");
    }

    switch (plan.schedule) {
    case Q4Q5AttnInputScheduleId::IndependentFixed:
        q4_rowsplit_execute_plan(plan.independent->query, x, q_weight, q, stream);
        q5_rowsplit_execute_plan(plan.independent->gate, x, gate_weight, gate, stream);
        q4_rowsplit_execute_plan(plan.independent->key, x, k_weight, k, stream);
        q5_rowsplit_execute_plan(plan.independent->value, x, v_weight, v, stream);
        return;
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR64C128:
        q4_q5_attn_input_grouped_mma_launch(plan.grouped_variant, x, q_weight, gate_weight,
                                            k_weight, v_weight, q, gate, k, v, stream);
        return;
    }
    throw std::logic_error("Q4/Q5 attention input: unknown schedule");
}

void q4_q5_attn_input_dispatch(const Tensor& x, const Weight& q_weight, const Weight& gate_weight,
                               const Weight& k_weight, const Weight& v_weight, Tensor& q,
                               Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], q_weight.padded_shape[1],
                                       x.ne[1]};
    const Q4Q5AttnInputPlan plan = q4_q5_attn_input_resolve_plan(problem);
    q4_q5_attn_input_execute_plan(plan, x, q_weight, gate_weight, k_weight, v_weight, q, gate, k, v,
                                  stream);
}

} // namespace ninfer::ops::detail
