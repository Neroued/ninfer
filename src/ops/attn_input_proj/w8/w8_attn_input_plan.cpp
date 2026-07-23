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

constexpr std::array<RouteSpec, 4> kRoutes{{
    {1, 1, W8AttnInputScheduleId::DecodeR8Direct},
    {2, 16, W8AttnInputScheduleId::SplitKMmaExactT},
    {17, 128, W8AttnInputScheduleId::MmaR32C128},
    {129, kAnyCols, W8AttnInputScheduleId::MmaR64C128},
}};

constexpr bool catalog_is_closed() {
    std::int64_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.first != expected || route.first > route.last) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(), "W8 attention input routes must be exact and closed");

bool supported_shape(const W8AttnInputProblem& problem) noexcept {
    return problem.input_rows == 2048 && problem.query_rows == 4096 && problem.kv_rows == 512 &&
           problem.parent_rows == 9216 && problem.padded_k == 2048;
}

W8KernelVariant variant_for(W8AttnInputScheduleId schedule, std::int32_t cols) {
    switch (schedule) {
    case W8AttnInputScheduleId::DecodeR8Direct:
    case W8AttnInputScheduleId::SplitKMmaExactT:
        return W8KernelVariant::None;
    case W8AttnInputScheduleId::SimtR8C4:
        return (cols % 4) == 0 ? W8KernelVariant::Full : W8KernelVariant::Predicated;
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
        return "attn_input_proj.w8.decode.r8.direct.k2048.split4";
    case W8AttnInputScheduleId::SplitKMmaExactT:
        return "attn_input_proj.w8.splitk8.mma.r16.exact_t.split4";
    case W8AttnInputScheduleId::SimtR8C4:
        return "attn_input_proj.w8.simt.r8.c4.split4";
    case W8AttnInputScheduleId::MmaR32C128:
        return "attn_input_proj.w8.mma.r32.c128.split4";
    case W8AttnInputScheduleId::MmaR64C128:
        return "attn_input_proj.w8.mma.r64.c128.split4";
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
    for (const RouteSpec& route : kRoutes) {
        if (problem.cols >= route.first && problem.cols <= route.last) {
            return {route.schedule, variant_for(route.schedule, problem.cols), 0};
        }
    }
    throw std::logic_error("W8 attention input: admitted problem has no covering route");
}

void w8_attn_input_execute_plan(const W8AttnInputPlan& plan, const Tensor& x, const Weight& weight,
                                Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                cudaStream_t stream) {
    const W8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    const W8AttnInputPlan resolved = w8_attn_input_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.variant != plan.variant ||
        resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("W8 attention input: plan does not match exact problem");
    }
    switch (plan.schedule) {
    case W8AttnInputScheduleId::DecodeR8Direct:
        w8_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
        return;
    case W8AttnInputScheduleId::SplitKMmaExactT:
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

} // namespace ninfer::ops::detail
