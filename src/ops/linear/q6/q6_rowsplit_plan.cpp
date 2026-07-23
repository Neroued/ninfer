#include "ops/linear/q6/q6_rowsplit_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct Q6ColsSet {
    std::int32_t first;
    std::int32_t last;
    std::int32_t step;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last && ((cols - first) % step) == 0;
    }
};

struct Q6SupportSpec {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
    Q6ColsSet admitted_cols;
    std::uint8_t route_begin;
    std::uint8_t route_count;
};

struct Q6RouteSpec {
    Q6ColsSet cols;
    Q6ScheduleId schedule;
};

constexpr std::array<Q6SupportSpec, 3> kSupportSpecs{{
    {248320, 5120, 5120, {1, kAnyCols, 1}, 0, 2},
    {248320, 2048, 2048, {1, kAnyCols, 1}, 2, 4},
    {1152, 1536, 1536, {4, 131072, 4}, 6, 9},
}};

constexpr std::array<Q6RouteSpec, 15> kRouteSpecs{{
    // [248320, 5120]
    {{1, 6, 1}, Q6ScheduleId::SimtR8C4},
    {{7, kAnyCols, 1}, Q6ScheduleId::MmaR64C128},

    // [248320, 2048]
    {{1, 4, 1}, Q6ScheduleId::SimtR8C4},
    {{5, 8, 1}, Q6ScheduleId::SimtR8C8},
    {{9, 64, 1}, Q6ScheduleId::MmaR64C64},
    {{65, kAnyCols, 1}, Q6ScheduleId::MmaR64C128},

    // [1152, 1536]
    {{4, 96, 4}, Q6ScheduleId::SimtR8C4},
    {{100, 704, 4}, Q6ScheduleId::MmaR64C64},
    {{708, 828, 4}, Q6ScheduleId::MmaR64C128},
    {{832, 832, 4}, Q6ScheduleId::MmaR64C64},
    {{836, 896, 4}, Q6ScheduleId::MmaR64C128},
    {{900, 960, 4}, Q6ScheduleId::MmaR64C64},
    {{964, 1024, 4}, Q6ScheduleId::MmaR64C128},
    {{1028, 1088, 4}, Q6ScheduleId::MmaR64C64},
    {{1092, 131072, 4}, Q6ScheduleId::MmaR64C128},
}};

constexpr bool known_schedule(Q6ScheduleId schedule) noexcept {
    switch (schedule) {
    case Q6ScheduleId::SimtR8C4:
    case Q6ScheduleId::SimtR8C8:
    case Q6ScheduleId::MmaR64C64:
    case Q6ScheduleId::MmaR64C128:
        return true;
    }
    return false;
}

constexpr bool catalog_is_closed() noexcept {
    std::size_t next_route = 0;
    for (std::size_t support_index = 0; support_index < kSupportSpecs.size(); ++support_index) {
        const Q6SupportSpec& support = kSupportSpecs[support_index];
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
            const Q6SupportSpec& other = kSupportSpecs[earlier];
            if (support.rows == other.rows && support.k == other.k &&
                support.padded_k == other.padded_k) {
                return false;
            }
        }

        std::int64_t expected_first = support.admitted_cols.first;
        for (std::size_t local = 0; local < support.route_count; ++local) {
            const Q6RouteSpec& route = kRouteSpecs[support.route_begin + local];
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
              "Q6 production admission and route spans must be exact, contiguous, and closed");

const Q6SupportSpec* find_support(const Q6Problem& problem) noexcept {
    for (const Q6SupportSpec& support : kSupportSpecs) {
        if (support.rows == problem.rows && support.k == problem.k &&
            support.padded_k == problem.padded_k) {
            return &support;
        }
    }
    return nullptr;
}

Q6KernelVariant resolve_variant(Q6ScheduleId schedule, const Q6Problem& problem) {
    switch (schedule) {
    case Q6ScheduleId::SimtR8C4:
    case Q6ScheduleId::SimtR8C8:
        if (q6_candidate_is_legal(schedule, Q6KernelVariant::None, problem)) {
            return Q6KernelVariant::None;
        }
        break;
    case Q6ScheduleId::MmaR64C64:
    case Q6ScheduleId::MmaR64C128:
        if (q6_candidate_is_legal(schedule, Q6KernelVariant::Full, problem)) {
            return Q6KernelVariant::Full;
        }
        if (q6_candidate_is_legal(schedule, Q6KernelVariant::Predicated, problem)) {
            return Q6KernelVariant::Predicated;
        }
        break;
    }
    throw std::logic_error("q6 linear: admitted route is not physically legal");
}

} // namespace

Q6Problem q6_rowsplit_problem(const Tensor& x, const Weight& w, const Tensor& out) noexcept {
    return {out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
}

bool q6_rowsplit_admits(const Q6Problem& problem) noexcept {
    const Q6SupportSpec* support = find_support(problem);
    return support != nullptr && support->admitted_cols.contains(problem.cols);
}

Q6Plan q6_rowsplit_resolve_plan(const Q6Problem& problem) {
    const Q6SupportSpec* support = find_support(problem);
    if (support == nullptr || !support->admitted_cols.contains(problem.cols)) {
        throw std::invalid_argument("q6 linear: exact problem or column count is not admitted");
    }

    for (std::size_t local = 0; local < support->route_count; ++local) {
        const Q6RouteSpec& route = kRouteSpecs[support->route_begin + local];
        if (route.cols.contains(problem.cols)) {
            return {route.schedule, resolve_variant(route.schedule, problem)};
        }
    }
    throw std::logic_error("q6 linear: admitted problem has no covering route");
}

void q6_rowsplit_execute_plan(Q6Plan plan, const Tensor& x, const Weight& w, Tensor& out,
                              cudaStream_t stream) {
    q6_rowsplit_launch_fixed(plan, x, w, out, stream);
}

void q6_rowsplit_dispatch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const Q6Plan plan = q6_rowsplit_resolve_plan(q6_rowsplit_problem(x, w, out));
    q6_rowsplit_execute_plan(plan, x, w, out, stream);
}

} // namespace ninfer::ops::detail
