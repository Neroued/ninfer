#include "ops/linear_add/w8/w8_linear_add_plan.h"

#include "ops/linear_add/w8/w8_linear_add_kernels.h"
#include "ops/common/token_slices.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    W8LinearAddScheduleId schedule;
};

constexpr std::array<RouteSpec, 5> kRoutes{{
    {1, 8, W8LinearAddScheduleId::SimtR8C4},
    {9, 16, W8LinearAddScheduleId::SplitKMmaExactT},
    {17, 52, W8LinearAddScheduleId::SimtR8C4},
    {53, 640, W8LinearAddScheduleId::MmaR32C128},
    {641, kAnyCols, W8LinearAddScheduleId::MmaR64C128},
}};

constexpr bool routes_are_closed() {
    std::int64_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.first != expected || route.last < route.first) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return kRoutes.back().last == kAnyCols && expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(routes_are_closed(), "W8 LinearAdd routes must be exact, contiguous, and closed");

W8ScheduleId base_schedule(W8LinearAddScheduleId schedule) {
    switch (schedule) {
    case W8LinearAddScheduleId::SimtR8C4:
        return W8ScheduleId::SimtR8C4;
    case W8LinearAddScheduleId::MmaR32C128:
        return W8ScheduleId::MmaR32C128;
    case W8LinearAddScheduleId::MmaR64C128:
        return W8ScheduleId::MmaR64C128;
    case W8LinearAddScheduleId::SplitKMmaExactT:
        break;
    }
    throw std::logic_error("w8 linear_add: exact-T schedule has no generic W8 schedule");
}

W8KernelVariant resolve_variant(W8LinearAddScheduleId schedule, const W8Problem& problem) {
    if (schedule == W8LinearAddScheduleId::SplitKMmaExactT) { return W8KernelVariant::None; }
    const W8ScheduleId base = base_schedule(schedule);
    if (w8_candidate_is_legal(base, W8KernelVariant::Full, problem)) {
        return W8KernelVariant::Full;
    }
    if (w8_candidate_is_legal(base, W8KernelVariant::Predicated, problem)) {
        return W8KernelVariant::Predicated;
    }
    throw std::logic_error("w8 linear_add: admitted route is not physically legal");
}

std::int32_t schedule_cols(W8LinearAddScheduleId schedule) {
    switch (schedule) {
    case W8LinearAddScheduleId::SimtR8C4:
        return 4;
    case W8LinearAddScheduleId::MmaR32C128:
    case W8LinearAddScheduleId::MmaR64C128:
        return 128;
    case W8LinearAddScheduleId::SplitKMmaExactT:
        break;
    }
    throw std::logic_error("w8 linear_add: exact-T schedule is not token-sliced");
}

} // namespace

const char* w8_linear_add_schedule_name(W8LinearAddScheduleId schedule) noexcept {
    switch (schedule) {
    case W8LinearAddScheduleId::SplitKMmaExactT:
        return "linear_add.w8.splitk8.mma.r16.exact_t.residual";
    case W8LinearAddScheduleId::SimtR8C4:
        return "linear_add.w8.simt.r8.c4.slab1024.s2.code_ca.scale_pair32";
    case W8LinearAddScheduleId::MmaR32C128:
        return "linear_add.w8.mma.r32.c128.k64.wr32.wc16.s2.scale_cache8.lb2";
    case W8LinearAddScheduleId::MmaR64C128:
        return "linear_add.w8.mma.r64.c128.k64.wr64.wc16.s2.scale_cache8.lb2";
    }
    return "linear_add.w8.unknown";
}

bool w8_linear_add_schedule_uses_mma(W8LinearAddScheduleId schedule) noexcept {
    return schedule == W8LinearAddScheduleId::SplitKMmaExactT ||
           schedule == W8LinearAddScheduleId::MmaR32C128 ||
           schedule == W8LinearAddScheduleId::MmaR64C128;
}

bool w8_linear_add_admits(const W8Problem& problem) noexcept {
    return problem.rows == 2048 && problem.k == 4096 && problem.padded_k == 4096 &&
           problem.cols >= 1;
}

W8LinearAddPlan w8_linear_add_resolve_plan(const W8Problem& problem) {
    if (!w8_linear_add_admits(problem)) {
        throw std::invalid_argument("w8 linear_add: exact problem or column count is not admitted");
    }
    for (const RouteSpec& route : kRoutes) {
        if (problem.cols >= route.first && problem.cols <= route.last) {
            return {route.schedule, resolve_variant(route.schedule, problem)};
        }
    }
    throw std::logic_error("w8 linear_add: admitted problem has no covering route");
}

void w8_linear_add_launch_candidate(W8ScheduleId schedule, W8KernelVariant variant, const Tensor& x,
                                    const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    const W8Problem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    if (!w8_linear_add_admits(problem) || !w8_candidate_is_legal(schedule, variant, problem)) {
        throw std::invalid_argument("w8 linear_add: illegal fixed candidate");
    }
    switch (schedule) {
    case W8ScheduleId::SimtR8C4:
        w8_linear_add_simt_r8_c4_launch(variant, x, w, residual_out, stream);
        return;
    case W8ScheduleId::SimtR8C8:
        w8_linear_add_simt_r8_c8_launch(variant, x, w, residual_out, stream);
        return;
    case W8ScheduleId::MmaR32C128:
        w8_linear_add_mma_r32_c128_launch(variant, x, w, residual_out, stream);
        return;
    case W8ScheduleId::MmaR64C128:
        w8_linear_add_mma_r64_c128_launch(variant, x, w, residual_out, stream);
        return;
    }
    throw std::logic_error("w8 linear_add: unknown schedule");
}

void w8_linear_add_execute_plan(const W8LinearAddPlan& plan, const Tensor& x, const Weight& w,
                                Tensor& residual_out, cudaStream_t stream) {
    const W8Problem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const W8LinearAddPlan resolved = w8_linear_add_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.variant != plan.variant) {
        throw std::invalid_argument("w8 linear_add: plan does not match the exact problem");
    }
    if (plan.schedule == W8LinearAddScheduleId::SplitKMmaExactT) {
        w8_linear_add_splitk_mma_launch(plan.variant, x, w, residual_out, stream);
        return;
    }
    const W8ScheduleId schedule = base_schedule(plan.schedule);
    for_each_token_slice(x.ne[1], schedule_cols(plan.schedule),
                         [&](std::int32_t offset, std::int32_t count) {
                             const Tensor x_slice  = x.slice(1, offset, count);
                             Tensor residual_slice = residual_out.slice(1, offset, count);
                             w8_linear_add_launch_candidate(schedule, plan.variant, x_slice, w,
                                                            residual_slice, stream);
                         });
}

void w8_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            cudaStream_t stream) {
    const W8Problem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    w8_linear_add_execute_plan(w8_linear_add_resolve_plan(problem), x, w, residual_out, stream);
}

} // namespace ninfer::ops::detail
