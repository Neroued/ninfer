#include "ops/linear/q5/q5_rowsplit_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kMaxTextCols = 128 * 65535;

struct Q5ColsSet {
    std::int32_t first;
    std::int32_t last;
    std::int32_t step;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last && ((cols - first) % step) == 0;
    }
};

struct Q5SupportSpec {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
    Q5ColsSet admitted_cols;
    std::uint8_t route_begin;
    std::uint8_t route_count;
};

struct Q5RouteSpec {
    Q5ColsSet cols;
    Q5ScheduleId schedule;
};

constexpr std::array<Q5SupportSpec, 6> kSupportSpecs{{
    {1024, 5120, 5120, {1, 16, 1}, 0, 2},
    {6144, 5120, 5120, {1, kMaxTextCols, 1}, 2, 5},
    {5120, 6144, 6144, {2, 24, 1}, 7, 2},
    {5120, 17408, 17408, {2, 24, 1}, 9, 2},
    {1152, 1152, 1152, {4, 131072, 4}, 11, 11},
    {1152, 4304, 4352, {4, 131072, 4}, 22, 3},
}};

constexpr std::array<Q5RouteSpec, 25> kRouteSpecs{{
    // [1024, 5120]
    {{1, 4, 1}, Q5ScheduleId::SimtR8C4},
    {{5, 16, 1}, Q5ScheduleId::SimtR8C8},

    // [6144, 5120]
    {{1, 1, 1}, Q5ScheduleId::GemvR16S2X},
    {{2, 6, 1}, Q5ScheduleId::SimtSplit4Exact},
    {{7, 24, 1}, Q5ScheduleId::SimtR8C8},
    {{25, 64, 1}, Q5ScheduleId::MmaR64C64},
    {{65, kMaxTextCols, 1}, Q5ScheduleId::MmaR64C128},

    // [5120, 6144]
    {{2, 6, 1}, Q5ScheduleId::SimtSplit2Exact},
    {{7, 24, 1}, Q5ScheduleId::SimtR8C8},

    // [5120, 17408]
    {{2, 6, 1}, Q5ScheduleId::SimtSplit2Exact},
    {{7, 24, 1}, Q5ScheduleId::SimtR8C8},

    // Vision [1152, 1152]
    {{4, 76, 4}, Q5ScheduleId::SimtR8C4},
    {{80, 636, 4}, Q5ScheduleId::MmaR64C64},
    {{640, 700, 4}, Q5ScheduleId::MmaR64C128},
    {{704, 704, 4}, Q5ScheduleId::MmaR64C64},
    {{708, 828, 4}, Q5ScheduleId::MmaR64C128},
    {{832, 832, 4}, Q5ScheduleId::MmaR64C64},
    {{836, 896, 4}, Q5ScheduleId::MmaR64C128},
    {{900, 960, 4}, Q5ScheduleId::MmaR64C64},
    {{964, 1024, 4}, Q5ScheduleId::MmaR64C128},
    {{1028, 1088, 4}, Q5ScheduleId::MmaR64C64},
    {{1092, 131072, 4}, Q5ScheduleId::MmaR64C128},

    // Vision [1152, 4304], Kpad=4352
    {{4, 120, 4}, Q5ScheduleId::SimtR8C4},
    {{124, 1148, 4}, Q5ScheduleId::MmaR64C64},
    {{1152, 131072, 4}, Q5ScheduleId::MmaR64C128},
}};

constexpr bool known_schedule(Q5ScheduleId schedule) noexcept {
    switch (schedule) {
    case Q5ScheduleId::GemvR16S2X:
    case Q5ScheduleId::SimtR8C4:
    case Q5ScheduleId::SimtR8C8:
    case Q5ScheduleId::SimtSplit2Exact:
    case Q5ScheduleId::SimtSplit4Exact:
    case Q5ScheduleId::MmaR64C64:
    case Q5ScheduleId::MmaR64C128:
        return true;
    }
    return false;
}

constexpr bool catalog_is_closed() noexcept {
    std::size_t next_route = 0;
    for (std::size_t support_index = 0; support_index < kSupportSpecs.size(); ++support_index) {
        const Q5SupportSpec& support = kSupportSpecs[support_index];
        if (support.rows <= 0 || support.k <= 0 || support.padded_k < support.k ||
            (support.padded_k % 128) != 0 || support.admitted_cols.first <= 0 ||
            support.admitted_cols.first > support.admitted_cols.last ||
            support.admitted_cols.step <= 0 || support.route_count == 0 ||
            support.route_begin != next_route ||
            static_cast<std::size_t>(support.route_begin) + support.route_count >
                kRouteSpecs.size()) {
            return false;
        }
        for (std::size_t earlier = 0; earlier < support_index; ++earlier) {
            const Q5SupportSpec& other = kSupportSpecs[earlier];
            if (support.rows == other.rows && support.k == other.k &&
                support.padded_k == other.padded_k) {
                return false;
            }
        }

        std::int32_t expected_first = support.admitted_cols.first;
        for (std::size_t local = 0; local < support.route_count; ++local) {
            const Q5RouteSpec& route = kRouteSpecs[support.route_begin + local];
            if (!known_schedule(route.schedule) || route.cols.first != expected_first ||
                route.cols.first > route.cols.last ||
                route.cols.step != support.admitted_cols.step ||
                ((route.cols.last - route.cols.first) % route.cols.step) != 0 ||
                !support.admitted_cols.contains(route.cols.first) ||
                !support.admitted_cols.contains(route.cols.last)) {
                return false;
            }
            expected_first = route.cols.last + route.cols.step;
        }
        if (expected_first != support.admitted_cols.last + support.admitted_cols.step) {
            return false;
        }
        next_route += support.route_count;
    }
    return next_route == kRouteSpecs.size();
}

static_assert(catalog_is_closed(),
              "Q5 production admission and route spans must be exact, contiguous, and closed");

const Q5SupportSpec* find_support(const Q5Problem& problem) noexcept {
    for (const Q5SupportSpec& support : kSupportSpecs) {
        if (support.rows == problem.rows && support.k == problem.k &&
            support.padded_k == problem.padded_k) {
            return &support;
        }
    }
    return nullptr;
}

Q5KernelVariant resolve_variant(Q5ScheduleId schedule, const Q5Problem& problem) {
    if (!q5_schedule_uses_mma(schedule)) {
        if (q5_candidate_is_legal(schedule, Q5KernelVariant::None, problem)) {
            return Q5KernelVariant::None;
        }
    } else {
        if (q5_candidate_is_legal(schedule, Q5KernelVariant::Full, problem)) {
            return Q5KernelVariant::Full;
        }
        if (q5_candidate_is_legal(schedule, Q5KernelVariant::Predicated, problem)) {
            return Q5KernelVariant::Predicated;
        }
    }
    throw std::logic_error("q5 linear: admitted route is not physically legal");
}

} // namespace

Q5Problem q5_rowsplit_problem(const Tensor& x, const Weight& w, const Tensor& out) noexcept {
    return {out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
}

bool q5_rowsplit_admits(const Q5Problem& problem) noexcept {
    const Q5SupportSpec* support = find_support(problem);
    return support != nullptr && support->admitted_cols.contains(problem.cols);
}

Q5Plan q5_rowsplit_resolve_plan(const Q5Problem& problem) {
    const Q5SupportSpec* support = find_support(problem);
    if (support == nullptr || !support->admitted_cols.contains(problem.cols)) {
        throw std::invalid_argument("q5 linear: exact problem or column count is not admitted");
    }

    for (std::size_t local = 0; local < support->route_count; ++local) {
        const Q5RouteSpec& route = kRouteSpecs[support->route_begin + local];
        if (route.cols.contains(problem.cols)) {
            return {route.schedule, resolve_variant(route.schedule, problem)};
        }
    }
    throw std::logic_error("q5 linear: admitted problem has no covering route");
}

void q5_rowsplit_execute_plan(Q5Plan plan, const Tensor& x, const Weight& w, Tensor& out,
                              cudaStream_t stream) {
    q5_rowsplit_launch_fixed(plan, x, w, out, stream);
}

void q5_rowsplit_dispatch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const Q5Plan plan = q5_rowsplit_resolve_plan(q5_rowsplit_problem(x, w, out));
    q5_rowsplit_execute_plan(plan, x, w, out, stream);
}

} // namespace ninfer::ops::detail
