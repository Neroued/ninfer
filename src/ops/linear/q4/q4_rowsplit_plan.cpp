#include "ops/linear/q4/q4_rowsplit_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct Q4ColsSet {
    std::int32_t first;
    std::int32_t last;
    std::int32_t step;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last && ((cols - first) % step) == 0;
    }
};

struct Q4SupportSpec {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
    Q4ColsSet admitted_cols;
    std::uint8_t route_begin;
    std::uint8_t route_count;
};

struct Q4RouteSpec {
    Q4ColsSet cols;
    Q4ScheduleId schedule;
};

constexpr std::array<Q4SupportSpec, 9> kSupportSpecs{{
    {1024, 5120, 5120, {1, kAnyCols, 1}, 0, 4},
    {4096, 5120, 5120, {1, kAnyCols, 1}, 4, 4},
    {6144, 5120, 5120, {1, kAnyCols, 1}, 8, 4},
    {7168, 5120, 5120, {1, kAnyCols, 1}, 12, 6},
    {34816, 5120, 5120, {1, kAnyCols, 1}, 18, 4},
    {131072, 5120, 5120, {1, kAnyCols, 1}, 22, 2},
    {131072, 2048, 2048, {1, kAnyCols, 1}, 24, 2},
    {3456, 1152, 1152, {4, 131072, 4}, 26, 3},
    {4304, 1152, 1152, {4, 131072, 4}, 29, 6},
}};

constexpr std::array<Q4RouteSpec, 35> kRouteSpecs{{
    // [1024, 5120]
    {{1, 1, 1}, Q4ScheduleId::GemvR1W8Direct},
    {{2, 15, 1}, Q4ScheduleId::SimtR8C4},
    {{16, 16, 1}, Q4ScheduleId::SimtR8C8},
    {{17, kAnyCols, 1}, Q4ScheduleId::MmaR64C128},

    // [4096, 5120]
    {{1, 1, 1}, Q4ScheduleId::GemvR1W8Direct},
    {{2, 4, 1}, Q4ScheduleId::SimtR8C4},
    {{5, 16, 1}, Q4ScheduleId::SimtR8C8},
    {{17, kAnyCols, 1}, Q4ScheduleId::MmaR64C128},

    // [6144, 5120]
    {{1, 1, 1}, Q4ScheduleId::GemvR1W8Direct},
    {{2, 7, 1}, Q4ScheduleId::SimtR8C4},
    {{8, 16, 1}, Q4ScheduleId::SimtR8C8},
    {{17, kAnyCols, 1}, Q4ScheduleId::MmaR64C128},

    // [7168, 5120]
    {{1, 1, 1}, Q4ScheduleId::GemvR1W8Direct},
    {{2, 7, 1}, Q4ScheduleId::SimtR8C4},
    {{8, 8, 1}, Q4ScheduleId::SimtR8C8},
    {{9, 15, 1}, Q4ScheduleId::SimtR8C4},
    {{16, 16, 1}, Q4ScheduleId::SimtR8C8},
    {{17, kAnyCols, 1}, Q4ScheduleId::MmaR64C128},

    // [34816, 5120]
    {{1, 1, 1}, Q4ScheduleId::GemvR1W8Direct},
    {{2, 4, 1}, Q4ScheduleId::SimtR8C4},
    {{5, 16, 1}, Q4ScheduleId::SimtR8C8},
    {{17, kAnyCols, 1}, Q4ScheduleId::MmaR64C128},

    // [131072, 5120]
    {{1, 1, 1}, Q4ScheduleId::GemvR4W1Direct},
    {{2, kAnyCols, 1}, Q4ScheduleId::MmaR64C128},

    // [131072, 2048]
    {{1, 1, 1}, Q4ScheduleId::GemvR4W1Direct},
    {{2, kAnyCols, 1}, Q4ScheduleId::MmaR64C128},

    // [3456, 1152]
    {{4, 36, 4}, Q4ScheduleId::SimtR8C4},
    {{40, 320, 4}, Q4ScheduleId::MmaR64C64},
    {{324, 131072, 4}, Q4ScheduleId::MmaR64C128},

    // [4304, 1152]
    {{4, 4, 4}, Q4ScheduleId::SimtR8C4},
    {{8, 8, 4}, Q4ScheduleId::SimtR8C8},
    {{12, 12, 4}, Q4ScheduleId::SimtR8C4},
    {{16, 24, 4}, Q4ScheduleId::SimtR8C8},
    {{28, 320, 4}, Q4ScheduleId::MmaR64C64},
    {{324, 131072, 4}, Q4ScheduleId::MmaR64C128},
}};

constexpr bool known_schedule(Q4ScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4ScheduleId::GemvR4W1Direct:
    case Q4ScheduleId::GemvR1W8Direct:
    case Q4ScheduleId::SimtR8C4:
    case Q4ScheduleId::SimtR8C8:
    case Q4ScheduleId::MmaR64C64:
    case Q4ScheduleId::MmaR64C128:
        return true;
    }
    return false;
}

constexpr bool catalog_is_closed() noexcept {
    std::size_t next_route = 0;
    for (std::size_t support_index = 0; support_index < kSupportSpecs.size(); ++support_index) {
        const Q4SupportSpec& support = kSupportSpecs[support_index];
        if (support.rows <= 0 || (support.rows % 16) != 0 || support.k <= 0 ||
            (support.k % 128) != 0 || support.padded_k != support.k ||
            support.admitted_cols.first <= 0 ||
            support.admitted_cols.first > support.admitted_cols.last ||
            support.admitted_cols.step <= 0 || support.route_count == 0 ||
            support.route_begin != next_route ||
            static_cast<std::size_t>(support.route_begin) + support.route_count >
                kRouteSpecs.size()) {
            return false;
        }
        for (std::size_t earlier = 0; earlier < support_index; ++earlier) {
            const Q4SupportSpec& other = kSupportSpecs[earlier];
            if (support.rows == other.rows && support.k == other.k &&
                support.padded_k == other.padded_k) {
                return false;
            }
        }

        std::int64_t expected_first = support.admitted_cols.first;
        for (std::size_t local = 0; local < support.route_count; ++local) {
            const Q4RouteSpec& route = kRouteSpecs[support.route_begin + local];
            if (!known_schedule(route.schedule) || route.cols.first != expected_first ||
                route.cols.first > route.cols.last ||
                route.cols.step != support.admitted_cols.step ||
                ((route.cols.last - route.cols.first) % route.cols.step) != 0 ||
                !support.admitted_cols.contains(route.cols.first) ||
                !support.admitted_cols.contains(route.cols.last)) {
                return false;
            }
            expected_first = static_cast<std::int64_t>(route.cols.last) + route.cols.step;
        }
        if (expected_first !=
            static_cast<std::int64_t>(support.admitted_cols.last) + support.admitted_cols.step) {
            return false;
        }
        next_route += support.route_count;
    }
    return next_route == kRouteSpecs.size();
}

static_assert(catalog_is_closed(),
              "Q4 production admission and route spans must be exact, contiguous, and closed");

const Q4SupportSpec* find_support(const Q4Problem& problem) noexcept {
    for (const Q4SupportSpec& support : kSupportSpecs) {
        if (support.rows == problem.rows && support.k == problem.k &&
            support.padded_k == problem.padded_k) {
            return &support;
        }
    }
    return nullptr;
}

Q4KernelVariant resolve_variant(Q4ScheduleId schedule, const Q4Problem& problem) {
    switch (schedule) {
    case Q4ScheduleId::GemvR4W1Direct:
    case Q4ScheduleId::GemvR1W8Direct:
        if (q4_candidate_is_legal(schedule, Q4KernelVariant::None, problem)) {
            return Q4KernelVariant::None;
        }
        break;
    case Q4ScheduleId::SimtR8C4:
    case Q4ScheduleId::SimtR8C8:
    case Q4ScheduleId::MmaR64C64:
    case Q4ScheduleId::MmaR64C128:
        if (q4_candidate_is_legal(schedule, Q4KernelVariant::Full, problem)) {
            return Q4KernelVariant::Full;
        }
        if (q4_candidate_is_legal(schedule, Q4KernelVariant::Predicated, problem)) {
            return Q4KernelVariant::Predicated;
        }
        break;
    }
    throw std::logic_error("q4 linear: admitted route is not physically legal");
}

} // namespace

Q4Problem q4_rowsplit_problem(const Tensor& x, const Weight& w, const Tensor& out) noexcept {
    return {out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
}

bool q4_rowsplit_admits(const Q4Problem& problem) noexcept {
    const Q4SupportSpec* support = find_support(problem);
    return support != nullptr && support->admitted_cols.contains(problem.cols);
}

Q4Plan q4_rowsplit_resolve_plan(const Q4Problem& problem) {
    const Q4SupportSpec* support = find_support(problem);
    if (support == nullptr || !support->admitted_cols.contains(problem.cols)) {
        throw std::invalid_argument("q4 linear: exact problem or column count is not admitted");
    }

    for (std::size_t local = 0; local < support->route_count; ++local) {
        const Q4RouteSpec& route = kRouteSpecs[support->route_begin + local];
        if (route.cols.contains(problem.cols)) {
            return {route.schedule, resolve_variant(route.schedule, problem)};
        }
    }
    throw std::logic_error("q4 linear: admitted problem has no covering route");
}

void q4_rowsplit_execute_plan(Q4Plan plan, const Tensor& x, const Weight& w, Tensor& out,
                              cudaStream_t stream) {
    q4_rowsplit_launch_fixed(plan, x, w, out, stream);
}

void q4_rowsplit_dispatch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const Q4Plan plan = q4_rowsplit_resolve_plan(q4_rowsplit_problem(x, w, out));
    q4_rowsplit_execute_plan(plan, x, w, out, stream);
}

} // namespace ninfer::ops::detail
