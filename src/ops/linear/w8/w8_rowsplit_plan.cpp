#include "ops/linear/w8/w8_rowsplit_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kMaxTextCols   = 128 * 65535;
constexpr std::int32_t kMaxVisionCols = 32768;

struct W8ColsSet {
    std::int32_t first;
    std::int32_t last;
    std::int32_t step;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last && ((cols - first) % step) == 0;
    }
};

struct W8SupportSpec {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
    W8ColsSet admitted_cols;
    std::uint8_t route_begin;
    std::uint8_t route_count;
};

struct W8RouteSpec {
    W8ColsSet cols;
    W8ScheduleId schedule;
};

constexpr std::array<W8SupportSpec, 9> kSupportSpecs{{
    {5120, 10240, 10240, {1, kMaxTextCols, 1}, 0, 3},
    {14336, 5120, 5120, {1, kMaxTextCols, 1}, 3, 3},
    {1024, 5120, 5120, {1, kMaxTextCols, 1}, 6, 3},
    {6144, 5120, 5120, {1, kMaxTextCols, 1}, 9, 3},
    {5120, 6144, 6144, {1, kMaxTextCols, 1}, 12, 3},
    {34816, 5120, 5120, {1, kMaxTextCols, 1}, 15, 3},
    {5120, 17408, 17408, {1, kMaxTextCols, 1}, 18, 3},
    {4608, 4608, 4608, {1, kMaxVisionCols, 1}, 21, 5},
    {5120, 4608, 4608, {1, kMaxVisionCols, 1}, 26, 3},
}};

constexpr std::array<W8RouteSpec, 29> kRouteSpecs{{
    // [5120,10240]
    {{1, 4, 1}, W8ScheduleId::SimtR8C4},
    {{5, 16, 1}, W8ScheduleId::SimtR8C8},
    {{17, kMaxTextCols, 1}, W8ScheduleId::MmaR64C128},

    // [14336,5120]
    {{1, 4, 1}, W8ScheduleId::SimtR8C4},
    {{5, 8, 1}, W8ScheduleId::SimtR8C8},
    {{9, kMaxTextCols, 1}, W8ScheduleId::MmaR64C128},

    // [1024,5120]
    {{1, 4, 1}, W8ScheduleId::SimtR8C4},
    {{5, 16, 1}, W8ScheduleId::SimtR8C8},
    {{17, kMaxTextCols, 1}, W8ScheduleId::MmaR32C128},

    // [6144,5120]
    {{1, 4, 1}, W8ScheduleId::SimtR8C4},
    {{5, 16, 1}, W8ScheduleId::SimtR8C8},
    {{17, kMaxTextCols, 1}, W8ScheduleId::MmaR64C128},

    // [5120,6144]
    {{1, 4, 1}, W8ScheduleId::SimtR8C4},
    {{5, 16, 1}, W8ScheduleId::SimtR8C8},
    {{17, kMaxTextCols, 1}, W8ScheduleId::MmaR64C128},

    // [34816,5120]
    {{1, 4, 1}, W8ScheduleId::SimtR8C4},
    {{5, 8, 1}, W8ScheduleId::SimtR8C8},
    {{9, kMaxTextCols, 1}, W8ScheduleId::MmaR64C128},

    // [5120,17408]
    {{1, 4, 1}, W8ScheduleId::SimtR8C4},
    {{5, 16, 1}, W8ScheduleId::SimtR8C8},
    {{17, kMaxTextCols, 1}, W8ScheduleId::MmaR64C128},

    // Vision merger [4608,4608]
    {{1, 8, 1}, W8ScheduleId::SimtR8C4},
    {{9, 11, 1}, W8ScheduleId::MmaR32C128},
    {{12, 12, 1}, W8ScheduleId::SimtR8C4},
    {{13, 256, 1}, W8ScheduleId::MmaR32C128},
    {{257, kMaxVisionCols, 1}, W8ScheduleId::MmaR64C128},

    // Vision merger [5120,4608]
    {{1, 4, 1}, W8ScheduleId::SimtR8C4},
    {{5, 5, 1}, W8ScheduleId::SimtR8C8},
    {{6, kMaxVisionCols, 1}, W8ScheduleId::MmaR64C128},
}};

constexpr bool known_schedule(W8ScheduleId schedule) noexcept {
    switch (schedule) {
    case W8ScheduleId::SimtR8C4:
    case W8ScheduleId::SimtR8C8:
    case W8ScheduleId::MmaR32C128:
    case W8ScheduleId::MmaR64C128:
        return true;
    }
    return false;
}

constexpr bool catalog_is_closed() noexcept {
    std::size_t next_route = 0;
    for (std::size_t support_index = 0; support_index < kSupportSpecs.size(); ++support_index) {
        const W8SupportSpec& support = kSupportSpecs[support_index];
        if (support.rows <= 0 || support.k <= 0 || support.padded_k != support.k ||
            (support.padded_k % 256) != 0 || support.admitted_cols.first <= 0 ||
            support.admitted_cols.first > support.admitted_cols.last ||
            support.admitted_cols.step <= 0 || support.route_count == 0 ||
            support.route_begin != next_route ||
            static_cast<std::size_t>(support.route_begin) + support.route_count >
                kRouteSpecs.size()) {
            return false;
        }
        for (std::size_t earlier = 0; earlier < support_index; ++earlier) {
            const W8SupportSpec& other = kSupportSpecs[earlier];
            if (support.rows == other.rows && support.k == other.k &&
                support.padded_k == other.padded_k) {
                return false;
            }
        }

        std::int32_t expected_first = support.admitted_cols.first;
        for (std::size_t local = 0; local < support.route_count; ++local) {
            const W8RouteSpec& route = kRouteSpecs[support.route_begin + local];
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
              "W8 production admission and route spans must be exact, contiguous, and closed");

const W8SupportSpec* find_support(const W8Problem& problem) noexcept {
    for (const W8SupportSpec& support : kSupportSpecs) {
        if (support.rows == problem.rows && support.k == problem.k &&
            support.padded_k == problem.padded_k) {
            return &support;
        }
    }
    return nullptr;
}

W8KernelVariant resolve_variant(W8ScheduleId schedule, const W8Problem& problem) {
    if (w8_candidate_is_legal(schedule, W8KernelVariant::Full, problem)) {
        return W8KernelVariant::Full;
    }
    if (w8_candidate_is_legal(schedule, W8KernelVariant::Predicated, problem)) {
        return W8KernelVariant::Predicated;
    }
    throw std::logic_error("w8 linear: admitted route is not physically legal");
}

} // namespace

W8Problem w8_rowsplit_problem(const Tensor& x, const Weight& w, const Tensor& out) noexcept {
    return {out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
}

bool w8_rowsplit_admits(const W8Problem& problem) noexcept {
    const W8SupportSpec* support = find_support(problem);
    return support != nullptr && support->admitted_cols.contains(problem.cols);
}

W8Plan w8_rowsplit_resolve_plan(const W8Problem& problem) {
    const W8SupportSpec* support = find_support(problem);
    if (support == nullptr || !support->admitted_cols.contains(problem.cols)) {
        throw std::invalid_argument("w8 linear: exact problem or column count is not admitted");
    }

    for (std::size_t local = 0; local < support->route_count; ++local) {
        const W8RouteSpec& route = kRouteSpecs[support->route_begin + local];
        if (route.cols.contains(problem.cols)) {
            return {route.schedule, resolve_variant(route.schedule, problem)};
        }
    }
    throw std::logic_error("w8 linear: admitted problem has no covering route");
}

void w8_rowsplit_execute_plan(W8Plan plan, const Tensor& x, const Weight& w, Tensor& out,
                              cudaStream_t stream) {
    w8_rowsplit_launch_fixed(plan, x, w, out, stream);
}

void w8_rowsplit_dispatch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const W8Plan plan = w8_rowsplit_resolve_plan(w8_rowsplit_problem(x, w, out));
    w8_rowsplit_execute_plan(plan, x, w, out, stream);
}

} // namespace ninfer::ops::detail
