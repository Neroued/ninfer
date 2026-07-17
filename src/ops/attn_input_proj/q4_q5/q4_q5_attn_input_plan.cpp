#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"

#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"
#include "ops/common/token_slices.h"

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
    Q4Q5AttnInputScheduleId schedule;
};

constexpr std::array<RouteSpec, 2> kRoutes{{
    {{1, 16}, Q4Q5AttnInputScheduleId::ParentSplitFixed},
    {{17, kAnyCols}, Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR64C128},
}};

constexpr bool catalog_is_closed() noexcept {
    return kRoutes[0].cols.first == 1 && kRoutes[0].cols.last + 1 == kRoutes[1].cols.first &&
           kRoutes[1].cols.last == kAnyCols;
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
    return same_q4_plan(lhs->query_key, rhs->query_key) &&
           same_q5_plan(lhs->gate_value, rhs->gate_value);
}

} // namespace

const char* q4_q5_attn_input_schedule_name(Q4Q5AttnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        return "attn_input_proj.q4_q5.parent_split_fixed";
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR64C128:
        return "attn_input_proj.q4_q5.grouped_homogeneous_pair.mma.r64.c128";
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
        Q4Q5AttnInputPlan plan{
            route.schedule,
            Q4KernelVariant::None,
            std::nullopt,
            0,
        };
        if (route.schedule == Q4Q5AttnInputScheduleId::ParentSplitFixed) {
            const std::int32_t parent_rows = problem.query_rows + problem.kv_rows;
            plan.parent_split              = Q4Q5AttnInputSubplans{
                q4_rowsplit_resolve_plan(
                    {parent_rows, problem.input_rows, problem.padded_k, problem.cols}),
                q5_rowsplit_resolve_plan(
                    {parent_rows, problem.input_rows, problem.padded_k, problem.cols}),
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
                                   const Weight& query_key_weight, const Weight& gate_value_weight,
                                   Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                   cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    const Q4Q5AttnInputPlan resolved = q4_q5_attn_input_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.grouped_variant != plan.grouped_variant ||
        !same_subplans(resolved.parent_split, plan.parent_split) ||
        resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("Q4/Q5 attention input: plan does not match exact problem");
    }

    switch (plan.schedule) {
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        q4_q5_attn_input_small_t_launch(plan.parent_split->query_key, plan.parent_split->gate_value,
                                        x, query_key_weight, gate_value_weight, q, gate, k, v,
                                        stream);
        return;
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR64C128:
        for_each_token_slice(problem.cols, 128, [&](std::int32_t offset, std::int32_t count) {
            const Tensor x_slice = x.slice(1, offset, count);
            Tensor q_slice       = q.slice(1, offset, count);
            Tensor gate_slice    = gate.slice(1, offset, count);
            Tensor k_slice       = k.slice(1, offset, count);
            Tensor v_slice       = v.slice(1, offset, count);
            q4_q5_attn_input_grouped_mma_launch(plan.grouped_variant, x_slice, query_key_weight,
                                                gate_value_weight, q_slice, gate_slice, k_slice,
                                                v_slice, stream);
        });
        return;
    }
    throw std::logic_error("Q4/Q5 attention input: unknown schedule");
}

void q4_q5_attn_input_dispatch(const Tensor& x, const Weight& query_key_weight,
                               const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k,
                               Tensor& v, cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    const Q4Q5AttnInputPlan plan = q4_q5_attn_input_resolve_plan(problem);
    q4_q5_attn_input_execute_plan(plan, x, query_key_weight, gate_value_weight, q, gate, k, v,
                                  stream);
}

} // namespace ninfer::ops::detail
