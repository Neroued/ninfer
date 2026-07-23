#include "ops/attn_input_proj/w8/w8_attn_input_plan.h"

#include "ops/attn_input_proj/w8/w8_attn_input_kernels.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    W8AttnInputScheduleId schedule;
};

constexpr std::array<RouteSpec, 4> kTargetRoutes{{
    {1, 1, W8AttnInputScheduleId::DecodeR8Direct},
    {2, 16, W8AttnInputScheduleId::SplitKMmaDirect},
    {17, 128, W8AttnInputScheduleId::MmaR32C128},
    {129, kAnyCols, W8AttnInputScheduleId::MmaR64C128},
}};

constexpr std::array<RouteSpec, 9> kCompanionRoutes{{
    {1, 1, W8AttnInputScheduleId::DecodeR8Direct},
    {2, 96, W8AttnInputScheduleId::SplitKMmaDirect},
    {97, 192, W8AttnInputScheduleId::MmaR32C64},
    {193, 288, W8AttnInputScheduleId::MmaR64C96},
    {289, 320, W8AttnInputScheduleId::MmaR64C64},
    {321, 384, W8AttnInputScheduleId::MmaR64C128},
    {385, 448, W8AttnInputScheduleId::MmaR128C64},
    {449, 560, W8AttnInputScheduleId::MmaR128C80},
    {561, kAnyCols, W8AttnInputScheduleId::MmaR64C128},
}};

template <std::size_t N>
constexpr bool catalog_is_closed(const std::array<RouteSpec, N>& routes) {
    std::int64_t expected = 1;
    for (const RouteSpec& route : routes) {
        if (route.first != expected || route.first > route.last) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(kTargetRoutes),
              "W8 target attention input routes must be exact and closed");
static_assert(catalog_is_closed(kCompanionRoutes),
              "W8 companion attention input routes must be exact and closed");

bool is_companion_shape(const W8AttnInputProblem& problem) noexcept {
    return problem.input_rows == 2048 && problem.query_rows == 4096 && problem.kv_rows == 1024 &&
           problem.parent_rows == 6144 && problem.padded_k == 2048;
}

bool supported_shape(const W8AttnInputProblem& problem) noexcept {
    const bool target_qkgv =
        problem.query_rows == 4096 && problem.kv_rows == 512 && problem.parent_rows == 9216;
    return problem.input_rows == 2048 && problem.padded_k == 2048 &&
           (target_qkgv || is_companion_shape(problem));
}

W8KernelVariant variant_for(W8AttnInputScheduleId schedule, std::int32_t cols) {
    switch (schedule) {
    case W8AttnInputScheduleId::DecodeR8Direct:
    case W8AttnInputScheduleId::SplitKMmaDirect:
        return W8KernelVariant::None;
    case W8AttnInputScheduleId::SimtR8C4:
        return (cols % 4) == 0 ? W8KernelVariant::Full : W8KernelVariant::Predicated;
    case W8AttnInputScheduleId::MmaR32C64:
    case W8AttnInputScheduleId::MmaR64C64:
    case W8AttnInputScheduleId::MmaR128C64:
        return (cols % 64) == 0 ? W8KernelVariant::Full : W8KernelVariant::Predicated;
    case W8AttnInputScheduleId::MmaR64C96:
        return (cols % 96) == 0 ? W8KernelVariant::Full : W8KernelVariant::Predicated;
    case W8AttnInputScheduleId::MmaR128C80:
        return (cols % 80) == 0 ? W8KernelVariant::Full : W8KernelVariant::Predicated;
    case W8AttnInputScheduleId::MmaR32C128:
    case W8AttnInputScheduleId::MmaR64C128:
        return (cols % 128) == 0 ? W8KernelVariant::Full : W8KernelVariant::Predicated;
    }
    throw std::logic_error("W8 attention input: unknown schedule");
}

} // namespace

const char* w8_attn_input_schedule_name(W8AttnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case W8AttnInputScheduleId::DecodeR8Direct:
        return "attn_input_proj.w8.decode.r8.direct.k2048";
    case W8AttnInputScheduleId::SplitKMmaDirect:
        return "attn_input_proj.w8.splitk.mma.r16.direct";
    case W8AttnInputScheduleId::SimtR8C4:
        return "attn_input_proj.w8.simt.r8.c4";
    case W8AttnInputScheduleId::MmaR32C64:
        return "attn_input_proj.w8.mma.r32.c64";
    case W8AttnInputScheduleId::MmaR32C128:
        return "attn_input_proj.w8.mma.r32.c128";
    case W8AttnInputScheduleId::MmaR64C64:
        return "attn_input_proj.w8.mma.r64.c64";
    case W8AttnInputScheduleId::MmaR64C96:
        return "attn_input_proj.w8.mma.r64.c96";
    case W8AttnInputScheduleId::MmaR64C128:
        return "attn_input_proj.w8.mma.r64.c128";
    case W8AttnInputScheduleId::MmaR128C64:
        return "attn_input_proj.w8.mma.r128.c64";
    case W8AttnInputScheduleId::MmaR128C80:
        return "attn_input_proj.w8.mma.r128.c80";
    }
    return "attn_input_proj.w8.unknown";
}

bool w8_attn_input_admits(const W8AttnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols > 0;
}

W8AttnInputPlan w8_attn_input_resolve_plan(const W8AttnInputProblem& problem) {
    if (!w8_attn_input_admits(problem)) {
        throw std::invalid_argument(
            "W8 attention input: exact problem or column count is not admitted");
    }
    const auto resolve_from = [&](const auto& routes) -> W8AttnInputPlan {
        for (const RouteSpec& route : routes) {
            if (problem.cols >= route.first && problem.cols <= route.last) {
                return {route.schedule, variant_for(route.schedule, problem.cols), 0};
            }
        }
        throw std::logic_error("W8 attention input: admitted problem has no covering route");
    };
    if (is_companion_shape(problem)) { return resolve_from(kCompanionRoutes); }
    return resolve_from(kTargetRoutes);
}

void w8_attn_input_execute_plan(const W8AttnInputPlan& plan, const Tensor& x, const Weight& weight,
                                Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                cudaStream_t stream) {
    const W8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    const W8AttnInputPlan resolved = w8_attn_input_resolve_plan(problem);
    if (problem.parent_rows != 9216 || problem.kv_rows != 512 ||
        resolved.schedule != plan.schedule || resolved.variant != plan.variant ||
        resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument(
            "W8 attention input: plan does not match exact four-output problem");
    }
    switch (plan.schedule) {
    case W8AttnInputScheduleId::DecodeR8Direct:
        w8_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
        return;
    case W8AttnInputScheduleId::SplitKMmaDirect:
        w8_attn_input_splitk_mma_launch(plan.variant, x, weight, q, gate, k, v, stream);
        return;
    case W8AttnInputScheduleId::SimtR8C4:
        w8_attn_input_simt_r8_c4_launch(plan.variant, x, weight, q, gate, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR32C128:
        w8_attn_input_mma_r32_c128_launch(plan.variant, x, weight, q, gate, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR64C128:
        w8_attn_input_mma_r64_c128_launch(plan.variant, x, weight, q, gate, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR32C64:
    case W8AttnInputScheduleId::MmaR64C64:
    case W8AttnInputScheduleId::MmaR64C96:
    case W8AttnInputScheduleId::MmaR128C64:
    case W8AttnInputScheduleId::MmaR128C80:
        throw std::logic_error("W8 attention input: companion schedule in four-output plan");
    }
    throw std::logic_error("W8 attention input: unknown schedule");
}

void w8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                            Tensor& k, Tensor& v, cudaStream_t stream) {
    const W8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    w8_attn_input_execute_plan(w8_attn_input_resolve_plan(problem), x, weight, q, gate, k, v,
                               stream);
}

void w8_attn_input_execute_plan(const W8AttnInputPlan& plan, const Tensor& x, const Weight& weight,
                                Tensor& q, Tensor& k, Tensor& v, cudaStream_t stream) {
    const W8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    const W8AttnInputPlan resolved = w8_attn_input_resolve_plan(problem);
    if (problem.parent_rows != 6144 || problem.kv_rows != 1024 ||
        resolved.schedule != plan.schedule || resolved.variant != plan.variant ||
        resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument(
            "W8 attention input: plan does not match exact three-output problem");
    }
    switch (plan.schedule) {
    case W8AttnInputScheduleId::DecodeR8Direct:
        w8_attn_input_decode_launch(x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::SplitKMmaDirect:
        w8_attn_input_splitk_mma_launch(plan.variant, x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::SimtR8C4:
        w8_attn_input_simt_r8_c4_launch(plan.variant, x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR32C128:
        w8_attn_input_mma_r32_c128_launch(plan.variant, x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR32C64:
        w8_companion_attn_input_mma_r32_c64_launch(plan.variant, x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR64C64:
        w8_companion_attn_input_mma_r64_c64_launch(plan.variant, x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR64C96:
        w8_companion_attn_input_mma_r64_c96_launch(plan.variant, x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR128C64:
        w8_companion_attn_input_mma_r128_c64_launch(plan.variant, x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR128C80:
        w8_companion_attn_input_mma_r128_c80_launch(plan.variant, x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR64C128:
        w8_attn_input_mma_r64_c128_launch(plan.variant, x, weight, q, k, v, stream);
        return;
    }
    throw std::logic_error("W8 attention input: unknown schedule");
}

void w8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                            cudaStream_t stream) {
    const W8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    w8_attn_input_execute_plan(w8_attn_input_resolve_plan(problem), x, weight, q, k, v, stream);
}

} // namespace ninfer::ops::detail
