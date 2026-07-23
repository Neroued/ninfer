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

constexpr std::array<RouteSpec, 5> kK4096Routes{{
    {1, 8, W8LinearAddScheduleId::SimtR8C4},
    {9, 16, W8LinearAddScheduleId::SplitKMmaExactT},
    {17, 52, W8LinearAddScheduleId::SimtR8C4},
    {53, 640, W8LinearAddScheduleId::MmaR32C128},
    {641, kAnyCols, W8LinearAddScheduleId::MmaR64C128},
}};

constexpr std::array<RouteSpec, 36> kK6144Routes{{
    {1, 1, W8LinearAddScheduleId::DecodeR16},
    {2, 32, W8LinearAddScheduleId::SplitKMmaExactT},
    {33, 65, W8LinearAddScheduleId::SplitKMma32PlusTail},
    {66, 95, W8LinearAddScheduleId::MmaR32C64},
    {96, 96, W8LinearAddScheduleId::MmaR32C96},
    {97, 128, W8LinearAddScheduleId::MmaR32C64},
    {129, 191, W8LinearAddScheduleId::MmaR32C128},
    {192, 192, W8LinearAddScheduleId::MmaR32C96},
    {193, 256, W8LinearAddScheduleId::MmaR32C128},
    {257, 384, W8LinearAddScheduleId::MmaR32C64},
    {385, 399, W8LinearAddScheduleId::MmaR32C96},
    {400, 400, W8LinearAddScheduleId::MmaR32C80},
    {401, 447, W8LinearAddScheduleId::MmaR32C96},
    {448, 448, W8LinearAddScheduleId::MmaR32C64},
    {449, 480, W8LinearAddScheduleId::MmaR32C96},
    {481, 640, W8LinearAddScheduleId::MmaR32C128},
    {641, 672, W8LinearAddScheduleId::MmaR48C96},
    {673, 704, W8LinearAddScheduleId::MmaR48C64},
    {705, 784, W8LinearAddScheduleId::MmaR48C112},
    {785, 896, W8LinearAddScheduleId::MmaR48C128},
    {897, 960, W8LinearAddScheduleId::MmaR64C96},
    {961, 1023, W8LinearAddScheduleId::MmaR64C112},
    {1024, 1024, W8LinearAddScheduleId::MmaR64C128},
    {1025, 1120, W8LinearAddScheduleId::MmaR64C112},
    {1121, 1280, W8LinearAddScheduleId::MmaR64C128},
    {1281, 1344, W8LinearAddScheduleId::MmaR128C64},
    {1345, 1408, W8LinearAddScheduleId::MmaR48C128},
    {1409, 1680, W8LinearAddScheduleId::MmaR128C80},
    {1681, 1791, W8LinearAddScheduleId::MmaR48C128},
    {1792, 1792, W8LinearAddScheduleId::MmaR64C128},
    {1793, 1919, W8LinearAddScheduleId::MmaR48C128},
    {1920, 1920, W8LinearAddScheduleId::MmaR64C128},
    {1921, 2016, W8LinearAddScheduleId::MmaR64C96},
    {2017, 2047, W8LinearAddScheduleId::MmaR64C112},
    {2048, 2048, W8LinearAddScheduleId::MmaR64C128},
    {2049, kAnyCols, W8LinearAddScheduleId::MmaR64C128},
}};

template <std::size_t N>
constexpr bool routes_are_closed(const std::array<RouteSpec, N>& routes) {
    std::int64_t expected = 1;
    for (const RouteSpec& route : routes) {
        if (route.first != expected || route.last < route.first) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return routes.back().last == kAnyCols && expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(routes_are_closed(kK4096Routes) && routes_are_closed(kK6144Routes),
              "W8 LinearAdd routes must be exact, contiguous, and closed");

std::int32_t schedule_rows(W8LinearAddScheduleId schedule) {
    switch (schedule) {
    case W8LinearAddScheduleId::DecodeR16:
    case W8LinearAddScheduleId::SplitKMma32PlusTail:
        break;
    case W8LinearAddScheduleId::SimtR8C4:
        return 8;
    case W8LinearAddScheduleId::MmaR32C64:
    case W8LinearAddScheduleId::MmaR32C80:
    case W8LinearAddScheduleId::MmaR32C96:
    case W8LinearAddScheduleId::MmaR32C128:
        return 32;
    case W8LinearAddScheduleId::MmaR48C64:
    case W8LinearAddScheduleId::MmaR48C96:
    case W8LinearAddScheduleId::MmaR48C112:
    case W8LinearAddScheduleId::MmaR48C128:
        return 48;
    case W8LinearAddScheduleId::MmaR64C96:
    case W8LinearAddScheduleId::MmaR64C112:
    case W8LinearAddScheduleId::MmaR64C128:
        return 64;
    case W8LinearAddScheduleId::MmaR128C64:
    case W8LinearAddScheduleId::MmaR128C80:
        return 128;
    case W8LinearAddScheduleId::SplitKMmaExactT:
        break;
    }
    throw std::logic_error("w8 linear_add: exact-T schedule has no row tile");
}

std::int32_t schedule_cols(W8LinearAddScheduleId schedule);

W8KernelVariant resolve_variant(W8LinearAddScheduleId schedule, const W8Problem& problem) {
    if (schedule == W8LinearAddScheduleId::DecodeR16 ||
        schedule == W8LinearAddScheduleId::SplitKMmaExactT ||
        schedule == W8LinearAddScheduleId::SplitKMma32PlusTail) {
        return W8KernelVariant::None;
    }
    return problem.rows % schedule_rows(schedule) == 0 &&
                   problem.cols % schedule_cols(schedule) == 0
               ? W8KernelVariant::Full
               : W8KernelVariant::Predicated;
}

std::int32_t schedule_cols(W8LinearAddScheduleId schedule) {
    switch (schedule) {
    case W8LinearAddScheduleId::DecodeR16:
    case W8LinearAddScheduleId::SplitKMma32PlusTail:
        break;
    case W8LinearAddScheduleId::SimtR8C4:
        return 4;
    case W8LinearAddScheduleId::MmaR32C64:
    case W8LinearAddScheduleId::MmaR48C64:
    case W8LinearAddScheduleId::MmaR128C64:
        return 64;
    case W8LinearAddScheduleId::MmaR32C80:
    case W8LinearAddScheduleId::MmaR128C80:
        return 80;
    case W8LinearAddScheduleId::MmaR32C96:
    case W8LinearAddScheduleId::MmaR48C96:
    case W8LinearAddScheduleId::MmaR64C96:
        return 96;
    case W8LinearAddScheduleId::MmaR48C112:
    case W8LinearAddScheduleId::MmaR64C112:
        return 112;
    case W8LinearAddScheduleId::MmaR32C128:
    case W8LinearAddScheduleId::MmaR48C128:
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
    case W8LinearAddScheduleId::DecodeR16:
        return "linear_add.w8.decode.r16.residual";
    case W8LinearAddScheduleId::SplitKMmaExactT:
        return "linear_add.w8.splitk8.mma.r16.exact_t.residual";
    case W8LinearAddScheduleId::SplitKMma32PlusTail:
        return "linear_add.w8.splitk8.mma.r16.exact32_plus_tail.residual";
    case W8LinearAddScheduleId::SimtR8C4:
        return "linear_add.w8.simt.r8.c4.slab1024.s2.code_ca.scale_pair32";
    case W8LinearAddScheduleId::MmaR32C64:
        return "linear_add.w8.mma.r32.c64.residual";
    case W8LinearAddScheduleId::MmaR32C80:
        return "linear_add.w8.mma.r32.c80.residual";
    case W8LinearAddScheduleId::MmaR32C96:
        return "linear_add.w8.mma.r32.c96.residual";
    case W8LinearAddScheduleId::MmaR32C128:
        return "linear_add.w8.mma.r32.c128.k64.wr32.wc16.s2.scale_cache8.lb2";
    case W8LinearAddScheduleId::MmaR48C64:
        return "linear_add.w8.mma.r48.c64.residual";
    case W8LinearAddScheduleId::MmaR48C96:
        return "linear_add.w8.mma.r48.c96.residual";
    case W8LinearAddScheduleId::MmaR48C112:
        return "linear_add.w8.mma.r48.c112.residual";
    case W8LinearAddScheduleId::MmaR48C128:
        return "linear_add.w8.mma.r48.c128.residual";
    case W8LinearAddScheduleId::MmaR64C96:
        return "linear_add.w8.mma.r64.c96.residual";
    case W8LinearAddScheduleId::MmaR64C112:
        return "linear_add.w8.mma.r64.c112.residual";
    case W8LinearAddScheduleId::MmaR64C128:
        return "linear_add.w8.mma.r64.c128.k64.wr64.wc16.s2.scale_cache8.lb2";
    case W8LinearAddScheduleId::MmaR128C64:
        return "linear_add.w8.mma.r128.c64.residual";
    case W8LinearAddScheduleId::MmaR128C80:
        return "linear_add.w8.mma.r128.c80.residual";
    }
    return "linear_add.w8.unknown";
}

bool w8_linear_add_schedule_uses_mma(W8LinearAddScheduleId schedule) noexcept {
    return schedule != W8LinearAddScheduleId::DecodeR16 &&
           schedule != W8LinearAddScheduleId::SimtR8C4;
}

bool w8_linear_add_admits(const W8Problem& problem) noexcept {
    return problem.rows == 2048 && (problem.k == 4096 || problem.k == 6144) &&
           problem.padded_k == problem.k && problem.cols >= 1;
}

W8LinearAddPlan w8_linear_add_resolve_plan(const W8Problem& problem) {
    if (!w8_linear_add_admits(problem)) {
        throw std::invalid_argument("w8 linear_add: exact problem or column count is not admitted");
    }
    const auto resolve_from = [&](const auto& routes) -> W8LinearAddPlan {
        for (const RouteSpec& route : routes) {
            if (problem.cols >= route.first && problem.cols <= route.last) {
                return {route.schedule, resolve_variant(route.schedule, problem)};
            }
        }
        throw std::logic_error("w8 linear_add: admitted problem has no covering route");
    };
    return problem.k == 6144 ? resolve_from(kK6144Routes) : resolve_from(kK4096Routes);
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
    if (plan.schedule == W8LinearAddScheduleId::DecodeR16) {
        w8_linear_add_decode_r16_launch(x, w, residual_out, stream);
        return;
    }
    if (plan.schedule == W8LinearAddScheduleId::SplitKMmaExactT) {
        w8_linear_add_splitk_mma_launch(plan.variant, x, w, residual_out, stream);
        return;
    }
    if (plan.schedule == W8LinearAddScheduleId::SplitKMma32PlusTail) {
        w8_linear_add_splitk_mma_composite_launch(x, w, residual_out, stream);
        return;
    }
    for_each_token_slice(
        x.ne[1], schedule_cols(plan.schedule), [&](std::int32_t offset, std::int32_t count) {
            const Tensor x_slice  = x.slice(1, offset, count);
            Tensor residual_slice = residual_out.slice(1, offset, count);
            switch (plan.schedule) {
            case W8LinearAddScheduleId::DecodeR16:
            case W8LinearAddScheduleId::SplitKMma32PlusTail:
                break;
            case W8LinearAddScheduleId::SimtR8C4:
                w8_linear_add_simt_r8_c4_launch(plan.variant, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR32C64:
                w8_linear_add_mma_r32_c64_launch(plan.variant, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR32C80:
                w8_linear_add_mma_r32_c80_launch(plan.variant, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR32C96:
                w8_linear_add_mma_r32_c96_launch(plan.variant, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR32C128:
                w8_linear_add_mma_r32_c128_launch(plan.variant, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR48C64:
                w8_linear_add_mma_r48_c64_launch(plan.variant, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR48C96:
                w8_linear_add_mma_r48_c96_launch(plan.variant, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR48C112:
                w8_linear_add_mma_r48_c112_launch(plan.variant, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR48C128:
                w8_linear_add_mma_r48_c128_launch(plan.variant, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR64C96:
                w8_linear_add_mma_r64_c96_launch(plan.variant, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR64C112:
                w8_linear_add_mma_r64_c112_launch(plan.variant, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR64C128:
                w8_linear_add_mma_r64_c128_launch(plan.variant, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR128C64:
                w8_linear_add_mma_r128_c64_launch(plan.variant, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR128C80:
                w8_linear_add_mma_r128_c80_launch(plan.variant, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::SplitKMmaExactT:
                break;
            }
            throw std::logic_error("w8 linear_add: unknown tiled schedule");
        });
}

void w8_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            cudaStream_t stream) {
    const W8Problem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    w8_linear_add_execute_plan(w8_linear_add_resolve_plan(problem), x, w, residual_out, stream);
}

} // namespace ninfer::ops::detail
