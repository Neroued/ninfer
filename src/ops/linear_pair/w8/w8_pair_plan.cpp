#include "ops/linear_pair/w8/w8_pair_plan.h"

#include "ops/linear_pair/w8/w8_pair_kernels.h"
#include "ops/common/token_slices.h"

#include <array>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct W8PairRouteSpec {
    std::int32_t first;
    std::int32_t last;
    W8PairScheduleId schedule;
};

constexpr std::array<W8PairRouteSpec, 3> kRoutes{{
    {1, 4, W8PairScheduleId::TwoSimtR8C4},
    {5, 56, W8PairScheduleId::TwoSimtR8C8},
    {57, kAnyCols, W8PairScheduleId::DualMmaR32C128},
}};

constexpr bool routes_are_closed() noexcept {
    std::int64_t expected = 1;
    for (const W8PairRouteSpec& route : kRoutes) {
        if (route.first != expected || route.last < route.first) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return kRoutes.back().last == kAnyCols && expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(routes_are_closed(), "W8 pair routes must be exact, contiguous, and closed");

W8ScheduleId base_schedule(W8PairScheduleId schedule) {
    switch (schedule) {
    case W8PairScheduleId::TwoSimtR8C4:
        return W8ScheduleId::SimtR8C4;
    case W8PairScheduleId::TwoSimtR8C8:
        return W8ScheduleId::SimtR8C8;
    case W8PairScheduleId::DualMmaR32C128:
        return W8ScheduleId::MmaR32C128;
    }
    throw std::logic_error("w8 pair: unknown schedule");
}

W8KernelVariant resolve_variant(W8PairScheduleId schedule, const W8PairProblem& problem) {
    const W8Problem base{problem.rows, problem.k, problem.padded_k, problem.cols};
    const W8ScheduleId candidate = base_schedule(schedule);
    if (w8_candidate_is_legal(candidate, W8KernelVariant::Full, base)) {
        return W8KernelVariant::Full;
    }
    if (w8_candidate_is_legal(candidate, W8KernelVariant::Predicated, base)) {
        return W8KernelVariant::Predicated;
    }
    throw std::logic_error("w8 pair: admitted route is not physically legal");
}

void require_pair_weights(const Weight& first_weight, const Weight& second_weight) {
    const auto valid = [](const Weight& w) {
        constexpr std::uint64_t kPayloadBytes = 5'570'560;
        return w.qtype == QType::W8G32_F16S && w.layout == QuantLayout::RowSplit &&
               w.scale_dtype == DType::FP16 && w.group == 32 && w.group_size == 32 && w.ndim == 2 &&
               w.n == 1024 && w.k == 5120 && w.shape[0] == 1024 && w.shape[1] == 5120 &&
               w.padded_shape[0] == 1024 && w.padded_shape[1] == 5120 &&
               w.payload_bytes >= kPayloadBytes && w.qdata != nullptr && w.qhigh == nullptr &&
               w.high_plane_bytes == 0 && w.scales != nullptr;
    };
    if (!valid(first_weight) || !valid(second_weight) || first_weight.n != second_weight.n ||
        first_weight.k != second_weight.k ||
        first_weight.padded_shape[1] != second_weight.padded_shape[1]) {
        throw std::invalid_argument("w8 pair: weights must be matching W8G32 RowSplit matrices");
    }
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_pair_operands(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                           const Tensor& first_out, const Tensor& second_out,
                           bool require_scale_16) {
    if (!aligned_to(x.data, 16) || !aligned_to(first_out.data, 16) ||
        !aligned_to(second_out.data, 16) || !aligned_to(first_weight.qdata, 16) ||
        !aligned_to(second_weight.qdata, 16) || !aligned_to(first_weight.scales, 4) ||
        !aligned_to(second_weight.scales, 4)) {
        throw std::invalid_argument(
            "w8 pair: requires 16-byte x/out/code and 4-byte scale alignment");
    }
    if (require_scale_16 &&
        (!aligned_to(first_weight.scales, 16) || !aligned_to(second_weight.scales, 16))) {
        throw std::invalid_argument("w8 pair MMA: scale planes must be 16-byte aligned");
    }
}

} // namespace

const char* w8_pair_schedule_name(W8PairScheduleId schedule) {
    switch (schedule) {
    case W8PairScheduleId::TwoSimtR8C4:
        return "w8_pair.two_simt.r8.c4";
    case W8PairScheduleId::TwoSimtR8C8:
        return "w8_pair.two_simt.r8.c8";
    case W8PairScheduleId::DualMmaR32C128:
        return "w8_pair.dual_mma.r32.c128";
    }
    return "w8_pair.unknown";
}

W8PairProblem w8_pair_problem(const Tensor& x, const Weight& first_weight,
                              const Tensor& first_out) noexcept {
    return {first_out.ne[0], x.ne[0], first_weight.padded_shape[1], x.ne[1]};
}

bool w8_pair_admits(const W8PairProblem& problem) noexcept {
    return problem.rows == 1024 && problem.k == 5120 && problem.padded_k == 5120 &&
           problem.cols >= 1;
}

W8PairPlan w8_pair_resolve_plan(const W8PairProblem& problem) {
    if (!w8_pair_admits(problem)) {
        throw std::invalid_argument("w8 pair: exact problem or column count is not admitted");
    }
    for (const W8PairRouteSpec& route : kRoutes) {
        if (problem.cols >= route.first && problem.cols <= route.last) {
            return {route.schedule, resolve_variant(route.schedule, problem), 0};
        }
    }
    throw std::logic_error("w8 pair: admitted problem has no covering route");
}

void w8_pair_execute_plan(W8PairPlan plan, const Tensor& x, const Weight& first_weight,
                          const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                          cudaStream_t stream) {
    require_pair_weights(first_weight, second_weight);
    const W8PairProblem problem = w8_pair_problem(x, first_weight, first_out);
    const W8PairPlan resolved   = w8_pair_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.variant != plan.variant ||
        resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("w8 pair: plan does not match the exact problem");
    }
    const W8Plan base_plan{
        plan.schedule == W8PairScheduleId::TwoSimtR8C4   ? W8ScheduleId::SimtR8C4
        : plan.schedule == W8PairScheduleId::TwoSimtR8C8 ? W8ScheduleId::SimtR8C8
                                                         : W8ScheduleId::MmaR32C128,
        plan.variant,
    };
    if (!w8_candidate_is_legal(base_plan.schedule, base_plan.variant,
                               {problem.rows, problem.k, problem.padded_k, problem.cols})) {
        throw std::invalid_argument(std::string("w8 pair fixed launch: illegal ") +
                                    w8_pair_schedule_name(plan.schedule) + "." +
                                    w8_kernel_variant_name(plan.variant));
    }
    require_pair_operands(x, first_weight, second_weight, first_out, second_out,
                          w8_schedule_uses_mma(base_plan.schedule));

    switch (plan.schedule) {
    case W8PairScheduleId::TwoSimtR8C4:
    case W8PairScheduleId::TwoSimtR8C8:
        w8_rowsplit_launch_fixed(base_plan, x, first_weight, first_out, stream);
        w8_rowsplit_launch_fixed(base_plan, x, second_weight, second_out, stream);
        return;
    case W8PairScheduleId::DualMmaR32C128:
        for_each_token_slice(problem.cols, 128, [&](std::int32_t offset, std::int32_t count) {
            const Tensor x_slice = x.slice(1, offset, count);
            Tensor first_slice   = first_out.slice(1, offset, count);
            Tensor second_slice  = second_out.slice(1, offset, count);
            w8_pair_gemm_mma_launch(plan.variant, x_slice, first_weight, second_weight, first_slice,
                                    second_slice, stream);
        });
        return;
    }
    throw std::logic_error("w8 pair fixed launch: unknown schedule");
}

void w8_pair_dispatch(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                      Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    const W8PairPlan plan = w8_pair_resolve_plan(w8_pair_problem(x, first_weight, first_out));
    w8_pair_execute_plan(plan, x, first_weight, second_weight, first_out, second_out, stream);
}

} // namespace ninfer::ops::detail
