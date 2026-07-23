#include "ops/linear_swiglu/w8/w8_linear_swiglu_plan.h"

#include "ops/linear_swiglu/w8/w8_linear_swiglu_kernels.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    W8LinearSwiGluScheduleId schedule;
};

constexpr std::array<RouteSpec, 18> kRoutes{{
    {1, 1, W8LinearSwiGluScheduleId::DecodePairR16},
    {2, 32, W8LinearSwiGluScheduleId::SplitKMmaExactT},
    {33, 64, W8LinearSwiGluScheduleId::MmaR32C64},
    {65, 80, W8LinearSwiGluScheduleId::MmaR32C80},
    {81, 96, W8LinearSwiGluScheduleId::MmaR32C96},
    {97, 128, W8LinearSwiGluScheduleId::MmaR64C64},
    {129, 192, W8LinearSwiGluScheduleId::MmaR32C64},
    {193, 240, W8LinearSwiGluScheduleId::MmaR128C80},
    {241, 255, W8LinearSwiGluScheduleId::MmaR32C128},
    {256, 256, W8LinearSwiGluScheduleId::MmaR64C128},
    {257, 264, W8LinearSwiGluScheduleId::MmaR64C64},
    {265, 288, W8LinearSwiGluScheduleId::MmaR64C96},
    {289, 320, W8LinearSwiGluScheduleId::MmaR64C64},
    {321, 384, W8LinearSwiGluScheduleId::MmaR64C128},
    {385, 448, W8LinearSwiGluScheduleId::MmaR128C64},
    {449, 512, W8LinearSwiGluScheduleId::MmaR64C128},
    {513, 560, W8LinearSwiGluScheduleId::MmaR128C80},
    {561, kAnyCols, W8LinearSwiGluScheduleId::MmaR64C128},
}};

constexpr bool catalog_is_closed() {
    std::int64_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.first != expected || route.first > route.last) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(), "W8 LinearSwiGLU routes must be exact and closed");

bool supported_shape(const W8LinearSwiGluProblem& problem) noexcept {
    return problem.gate_up_rows == 12288 && problem.output_rows == 6144 && problem.k == 2048 &&
           problem.padded_k == 2048;
}

W8KernelVariant variant_for(W8LinearSwiGluScheduleId schedule, std::int32_t cols) {
    switch (schedule) {
    case W8LinearSwiGluScheduleId::DecodePairR16:
    case W8LinearSwiGluScheduleId::SplitKMmaExactT:
        return W8KernelVariant::None;
    case W8LinearSwiGluScheduleId::MmaR32C64:
    case W8LinearSwiGluScheduleId::MmaR64C64:
    case W8LinearSwiGluScheduleId::MmaR128C64:
        return (cols % 64) == 0 ? W8KernelVariant::Full : W8KernelVariant::Predicated;
    case W8LinearSwiGluScheduleId::MmaR32C80:
    case W8LinearSwiGluScheduleId::MmaR128C80:
        return (cols % 80) == 0 ? W8KernelVariant::Full : W8KernelVariant::Predicated;
    case W8LinearSwiGluScheduleId::MmaR32C96:
    case W8LinearSwiGluScheduleId::MmaR64C96:
        return (cols % 96) == 0 ? W8KernelVariant::Full : W8KernelVariant::Predicated;
    case W8LinearSwiGluScheduleId::MmaR32C128:
    case W8LinearSwiGluScheduleId::MmaR64C128:
        return (cols % 128) == 0 ? W8KernelVariant::Full : W8KernelVariant::Predicated;
    }
    throw std::logic_error("W8 LinearSwiGLU: unknown schedule");
}

} // namespace

const char* w8_linear_swiglu_schedule_name(W8LinearSwiGluScheduleId schedule) noexcept {
    switch (schedule) {
    case W8LinearSwiGluScheduleId::DecodePairR16:
        return "linear_swiglu.w8.decode.pair.r16";
    case W8LinearSwiGluScheduleId::SplitKMmaExactT:
        return "linear_swiglu.w8.splitk.mma.pair.r8.exact_t";
    case W8LinearSwiGluScheduleId::MmaR32C64:
        return "linear_swiglu.w8.mma.pair.r16.c64";
    case W8LinearSwiGluScheduleId::MmaR32C80:
        return "linear_swiglu.w8.mma.pair.r16.c80";
    case W8LinearSwiGluScheduleId::MmaR32C96:
        return "linear_swiglu.w8.mma.pair.r16.c96";
    case W8LinearSwiGluScheduleId::MmaR32C128:
        return "linear_swiglu.w8.mma.pair.r16.c128";
    case W8LinearSwiGluScheduleId::MmaR64C64:
        return "linear_swiglu.w8.mma.pair.r32.c64";
    case W8LinearSwiGluScheduleId::MmaR64C96:
        return "linear_swiglu.w8.mma.pair.r32.c96";
    case W8LinearSwiGluScheduleId::MmaR64C128:
        return "linear_swiglu.w8.mma.pair.r32.c128";
    case W8LinearSwiGluScheduleId::MmaR128C64:
        return "linear_swiglu.w8.mma.pair.r64.c64";
    case W8LinearSwiGluScheduleId::MmaR128C80:
        return "linear_swiglu.w8.mma.pair.r64.c80";
    }
    return "linear_swiglu.w8.unknown";
}

bool w8_linear_swiglu_schedule_uses_mma(W8LinearSwiGluScheduleId schedule) noexcept {
    return schedule != W8LinearSwiGluScheduleId::DecodePairR16;
}

bool w8_linear_swiglu_admits(const W8LinearSwiGluProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols > 0;
}

W8LinearSwiGluPlan w8_linear_swiglu_resolve_plan(const W8LinearSwiGluProblem& problem) {
    if (!w8_linear_swiglu_admits(problem)) {
        throw std::invalid_argument(
            "W8 LinearSwiGLU: exact problem or column count is not admitted");
    }
    for (const RouteSpec& route : kRoutes) {
        if (problem.cols >= route.first && problem.cols <= route.last) {
            return {route.schedule, variant_for(route.schedule, problem.cols), 0};
        }
    }
    throw std::logic_error("W8 LinearSwiGLU: admitted problem has no route");
}

void w8_linear_swiglu_execute_plan(const W8LinearSwiGluPlan& plan, const Tensor& x, const Weight& w,
                                   Tensor& out, cudaStream_t stream) {
    const W8LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const W8LinearSwiGluPlan resolved = w8_linear_swiglu_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.variant != plan.variant ||
        resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("W8 LinearSwiGLU: plan does not match exact problem");
    }
    switch (plan.schedule) {
    case W8LinearSwiGluScheduleId::DecodePairR16:
        w8_linear_swiglu_decode_pair_r16_launch(x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::SplitKMmaExactT:
        w8_linear_swiglu_splitk_exact_t_launch(plan.variant, x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR32C64:
        w8_linear_swiglu_mma_r32_c64_launch(plan.variant, x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR32C80:
        w8_linear_swiglu_mma_r32_c80_launch(plan.variant, x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR32C96:
        w8_linear_swiglu_mma_r32_c96_launch(plan.variant, x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR32C128:
        w8_linear_swiglu_mma_r32_c128_launch(plan.variant, x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR64C64:
        w8_linear_swiglu_mma_r64_c64_launch(plan.variant, x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR64C96:
        w8_linear_swiglu_mma_r64_c96_launch(plan.variant, x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR64C128:
        w8_linear_swiglu_mma_r64_c128_launch(plan.variant, x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR128C64:
        w8_linear_swiglu_mma_r128_c64_launch(plan.variant, x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR128C80:
        w8_linear_swiglu_mma_r128_c80_launch(plan.variant, x, w, out, stream);
        return;
    }
    throw std::logic_error("W8 LinearSwiGLU: unknown schedule");
}

void w8_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const W8LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    w8_linear_swiglu_execute_plan(w8_linear_swiglu_resolve_plan(problem), x, w, out, stream);
}

} // namespace ninfer::ops::detail
