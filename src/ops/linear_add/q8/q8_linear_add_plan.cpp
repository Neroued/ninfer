#include "core/weight.h"
#include "ops/linear_add/q8/q8_linear_add_plan.h"

#include "ops/linear_add/q8/q8_linear_add_kernels.h"
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
    Q8LinearAddScheduleId schedule;
};

constexpr std::array<RouteSpec, 5> kK4096Routes{{
    {1, 1, Q8LinearAddScheduleId::SimtR8C4},
    {2, 48, Q8LinearAddScheduleId::SplitKMmaExactT},
    {49, 128, Q8LinearAddScheduleId::MediumSplitK},
    {129, 640, Q8LinearAddScheduleId::MmaR32C128},
    {641, kAnyCols, Q8LinearAddScheduleId::MmaR64C128},
}};

constexpr std::array<RouteSpec, 33> kK6144Routes{{
    {1, 1, Q8LinearAddScheduleId::DecodeR16},
    {2, 48, Q8LinearAddScheduleId::SplitKMmaExactT},
    {49, 128, Q8LinearAddScheduleId::MediumSplitK},
    {129, 191, Q8LinearAddScheduleId::MmaR32C128},
    {192, 192, Q8LinearAddScheduleId::MmaR32C96},
    {193, 256, Q8LinearAddScheduleId::MmaR32C128},
    {257, 384, Q8LinearAddScheduleId::MmaR32C64},
    {385, 399, Q8LinearAddScheduleId::MmaR32C96},
    {400, 400, Q8LinearAddScheduleId::MmaR32C80},
    {401, 447, Q8LinearAddScheduleId::MmaR32C96},
    {448, 448, Q8LinearAddScheduleId::MmaR32C64},
    {449, 480, Q8LinearAddScheduleId::MmaR32C96},
    {481, 640, Q8LinearAddScheduleId::MmaR32C128},
    {641, 672, Q8LinearAddScheduleId::MmaR48C96},
    {673, 704, Q8LinearAddScheduleId::MmaR48C64},
    {705, 784, Q8LinearAddScheduleId::MmaR48C112},
    {785, 896, Q8LinearAddScheduleId::MmaR48C128},
    {897, 960, Q8LinearAddScheduleId::MmaR64C96},
    {961, 1023, Q8LinearAddScheduleId::MmaR64C112},
    {1024, 1024, Q8LinearAddScheduleId::MmaR64C128},
    {1025, 1120, Q8LinearAddScheduleId::MmaR64C112},
    {1121, 1280, Q8LinearAddScheduleId::MmaR64C128},
    {1281, 1344, Q8LinearAddScheduleId::MmaR128C64},
    {1345, 1408, Q8LinearAddScheduleId::MmaR48C128},
    {1409, 1680, Q8LinearAddScheduleId::MmaR128C80},
    {1681, 1791, Q8LinearAddScheduleId::MmaR48C128},
    {1792, 1792, Q8LinearAddScheduleId::MmaR64C128},
    {1793, 1919, Q8LinearAddScheduleId::MmaR48C128},
    {1920, 1920, Q8LinearAddScheduleId::MmaR64C128},
    {1921, 2016, Q8LinearAddScheduleId::MmaR64C96},
    {2017, 2047, Q8LinearAddScheduleId::MmaR64C112},
    {2048, 2048, Q8LinearAddScheduleId::MmaR64C128},
    {2049, kAnyCols, Q8LinearAddScheduleId::MmaR64C128},
}};

constexpr std::array<RouteSpec, 2> kN5120Routes{{
    {1, 64, Q8LinearAddScheduleId::SplitKMmaCapacity},
    {65, kAnyCols, Q8LinearAddScheduleId::GroupedSplitK},
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

static_assert(routes_are_closed(kK4096Routes) && routes_are_closed(kK6144Routes) &&
                  routes_are_closed(kN5120Routes),
              "Q8 LinearAdd routes must be exact, contiguous, and closed");

std::int32_t schedule_rows(Q8LinearAddScheduleId schedule) {
    switch (schedule) {
    case Q8LinearAddScheduleId::DecodeR16:
    case Q8LinearAddScheduleId::MediumSplitK:
        break;
    case Q8LinearAddScheduleId::SimtR8C4:
        return 8;
    case Q8LinearAddScheduleId::MmaR32C64:
    case Q8LinearAddScheduleId::MmaR32C80:
    case Q8LinearAddScheduleId::MmaR32C96:
    case Q8LinearAddScheduleId::MmaR32C128:
        return 32;
    case Q8LinearAddScheduleId::MmaR48C64:
    case Q8LinearAddScheduleId::MmaR48C96:
    case Q8LinearAddScheduleId::MmaR48C112:
    case Q8LinearAddScheduleId::MmaR48C128:
        return 48;
    case Q8LinearAddScheduleId::MmaR64C96:
    case Q8LinearAddScheduleId::MmaR64C112:
    case Q8LinearAddScheduleId::MmaR64C128:
        return 64;
    case Q8LinearAddScheduleId::MmaR128C64:
    case Q8LinearAddScheduleId::MmaR128C80:
        return 128;
    case Q8LinearAddScheduleId::SplitKMmaExactT:
    case Q8LinearAddScheduleId::SplitKMmaCapacity:
    case Q8LinearAddScheduleId::GroupedSplitK:
        break;
    }
    throw std::logic_error("q8 linear_add: exact-T schedule has no row tile");
}

std::int32_t schedule_cols(Q8LinearAddScheduleId schedule);

bool use_full(Q8LinearAddScheduleId schedule, const Q8LinearAddProblem& problem) {
    return problem.rows % schedule_rows(schedule) == 0 &&
           problem.cols % schedule_cols(schedule) == 0;
}

std::int32_t schedule_cols(Q8LinearAddScheduleId schedule) {
    switch (schedule) {
    case Q8LinearAddScheduleId::DecodeR16:
    case Q8LinearAddScheduleId::MediumSplitK:
        break;
    case Q8LinearAddScheduleId::SimtR8C4:
        return 4;
    case Q8LinearAddScheduleId::MmaR32C64:
    case Q8LinearAddScheduleId::MmaR48C64:
    case Q8LinearAddScheduleId::MmaR128C64:
        return 64;
    case Q8LinearAddScheduleId::MmaR32C80:
    case Q8LinearAddScheduleId::MmaR128C80:
        return 80;
    case Q8LinearAddScheduleId::MmaR32C96:
    case Q8LinearAddScheduleId::MmaR48C96:
    case Q8LinearAddScheduleId::MmaR64C96:
        return 96;
    case Q8LinearAddScheduleId::MmaR48C112:
    case Q8LinearAddScheduleId::MmaR64C112:
        return 112;
    case Q8LinearAddScheduleId::MmaR32C128:
    case Q8LinearAddScheduleId::MmaR48C128:
    case Q8LinearAddScheduleId::MmaR64C128:
        return 128;
    case Q8LinearAddScheduleId::SplitKMmaExactT:
    case Q8LinearAddScheduleId::SplitKMmaCapacity:
    case Q8LinearAddScheduleId::GroupedSplitK:
        break;
    }
    throw std::logic_error("q8 linear_add: exact-T schedule is not token-sliced");
}

} // namespace

const char* q8_linear_add_schedule_name(Q8LinearAddScheduleId schedule) noexcept {
    switch (schedule) {
    case Q8LinearAddScheduleId::DecodeR16:
        return "linear_add.q8.decode.r16.residual";
    case Q8LinearAddScheduleId::GroupedSplitK:
        return "linear_add.q8.grouped_splitk.residual";
    case Q8LinearAddScheduleId::SplitKMmaCapacity:
        return "linear_add.q8.splitk.mma.capacity.residual";
    case Q8LinearAddScheduleId::SplitKMmaExactT:
        return "linear_add.q8.splitk8.mma.r16.exact_t.residual";
    case Q8LinearAddScheduleId::MediumSplitK:
        return "linear_add.q8.medium_splitk.residual";
    case Q8LinearAddScheduleId::SimtR8C4:
        return "linear_add.q8.simt.r8.c4.slab1024.s2.code_ca.scale_pair32";
    case Q8LinearAddScheduleId::MmaR32C64:
        return "linear_add.q8.mma.r32.c64.residual";
    case Q8LinearAddScheduleId::MmaR32C80:
        return "linear_add.q8.mma.r32.c80.residual";
    case Q8LinearAddScheduleId::MmaR32C96:
        return "linear_add.q8.mma.r32.c96.residual";
    case Q8LinearAddScheduleId::MmaR32C128:
        return "linear_add.q8.mma.r32.c128.k64.wr32.wc16.s2.scale_cache8.lb2";
    case Q8LinearAddScheduleId::MmaR48C64:
        return "linear_add.q8.mma.r48.c64.residual";
    case Q8LinearAddScheduleId::MmaR48C96:
        return "linear_add.q8.mma.r48.c96.residual";
    case Q8LinearAddScheduleId::MmaR48C112:
        return "linear_add.q8.mma.r48.c112.residual";
    case Q8LinearAddScheduleId::MmaR48C128:
        return "linear_add.q8.mma.r48.c128.residual";
    case Q8LinearAddScheduleId::MmaR64C96:
        return "linear_add.q8.mma.r64.c96.residual";
    case Q8LinearAddScheduleId::MmaR64C112:
        return "linear_add.q8.mma.r64.c112.residual";
    case Q8LinearAddScheduleId::MmaR64C128:
        return "linear_add.q8.mma.r64.c128.k64.wr64.wc16.s2.scale_cache8.lb2";
    case Q8LinearAddScheduleId::MmaR128C64:
        return "linear_add.q8.mma.r128.c64.residual";
    case Q8LinearAddScheduleId::MmaR128C80:
        return "linear_add.q8.mma.r128.c80.residual";
    }
    return "linear_add.q8.unknown";
}

bool q8_linear_add_schedule_uses_mma(Q8LinearAddScheduleId schedule) noexcept {
    return schedule != Q8LinearAddScheduleId::DecodeR16 &&
           schedule != Q8LinearAddScheduleId::SimtR8C4;
}

bool q8_linear_add_admits(const Q8LinearAddProblem& problem) noexcept {
    const bool shape = (problem.rows == 2048 && (problem.k == 4096 || problem.k == 6144)) ||
                       (problem.rows == 5120 && (problem.k == 6144 || problem.k == 17408));
    return shape && problem.padded_k == problem.k && problem.cols >= 1;
}

Q8LinearAddPlan q8_linear_add_resolve_plan(const Q8LinearAddProblem& problem) {
    if (!q8_linear_add_admits(problem)) {
        throw std::invalid_argument("q8 linear_add: exact problem or column count is not admitted");
    }
    const auto resolve_from = [&](const auto& routes) -> Q8LinearAddPlan {
        for (const RouteSpec& route : routes) {
            if (problem.cols >= route.first && problem.cols <= route.last) {
                return {route.schedule};
            }
        }
        throw std::logic_error("q8 linear_add: admitted problem has no covering route");
    };
    if (problem.rows == 5120) { return resolve_from(kN5120Routes); }
    return problem.k == 6144 ? resolve_from(kK6144Routes) : resolve_from(kK4096Routes);
}

void q8_linear_add_execute_plan(const Q8LinearAddPlan& plan, const Tensor& x, const Weight& w,
                                Tensor& residual_out, cudaStream_t stream) {
    const Q8LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q8LinearAddPlan resolved = q8_linear_add_resolve_plan(problem);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument("q8 linear_add: plan does not match the exact problem");
    }
    if (plan.schedule == Q8LinearAddScheduleId::GroupedSplitK) {
        q8_linear_add_grouped_launch(x, w, residual_out, stream);
        return;
    }
    if (plan.schedule == Q8LinearAddScheduleId::SplitKMmaCapacity) {
        q8_linear_add_splitk_capacity_launch(x, w, residual_out, stream);
        return;
    }
    if (plan.schedule == Q8LinearAddScheduleId::DecodeR16) {
        q8_linear_add_decode_r16_launch(x, w, residual_out, stream);
        return;
    }
    if (plan.schedule == Q8LinearAddScheduleId::SplitKMmaExactT) {
        q8_linear_add_splitk_mma_launch(x, w, residual_out, stream);
        return;
    }
    if (plan.schedule == Q8LinearAddScheduleId::MediumSplitK) {
        q8_linear_add_medium_splitk_launch(x, w, residual_out, stream);
        return;
    }
    const bool full = use_full(plan.schedule, problem);
    for_each_token_slice(
        x.ne[1], schedule_cols(plan.schedule), [&](std::int32_t offset, std::int32_t count) {
            const Tensor x_slice  = x.slice(1, offset, count);
            Tensor residual_slice = residual_out.slice(1, offset, count);
            switch (plan.schedule) {
            case Q8LinearAddScheduleId::DecodeR16:
            case Q8LinearAddScheduleId::MediumSplitK:
                break;
            case Q8LinearAddScheduleId::SimtR8C4:
                q8_linear_add_simt_r8_c4_launch(full, x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR32C64:
                q8_linear_add_mma_r32_c64_launch(full, x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR32C80:
                q8_linear_add_mma_r32_c80_launch(full, x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR32C96:
                q8_linear_add_mma_r32_c96_launch(full, x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR32C128:
                q8_linear_add_mma_r32_c128_launch(full, x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR48C64:
                q8_linear_add_mma_r48_c64_launch(full, x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR48C96:
                q8_linear_add_mma_r48_c96_launch(full, x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR48C112:
                q8_linear_add_mma_r48_c112_launch(full, x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR48C128:
                q8_linear_add_mma_r48_c128_launch(full, x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR64C96:
                q8_linear_add_mma_r64_c96_launch(full, x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR64C112:
                q8_linear_add_mma_r64_c112_launch(full, x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR64C128:
                q8_linear_add_mma_r64_c128_launch(full, x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR128C64:
                q8_linear_add_mma_r128_c64_launch(full, x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR128C80:
                q8_linear_add_mma_r128_c80_launch(full, x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::GroupedSplitK:
            case Q8LinearAddScheduleId::SplitKMmaCapacity:
            case Q8LinearAddScheduleId::SplitKMmaExactT:
                break;
            }
            throw std::logic_error("q8 linear_add: unknown tiled schedule");
        });
}

void q8_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            cudaStream_t stream) {
    const Q8LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    q8_linear_add_execute_plan(q8_linear_add_resolve_plan(problem), x, w, residual_out, stream);
}

} // namespace ninfer::ops::detail
