#include "ninfer/ops/linear_swiglu.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

using ninfer::ops::detail::Q4LinearSwiGluPlan;
using ninfer::ops::detail::Q4LinearSwiGluProblem;
using ninfer::ops::detail::Q4LinearSwiGluScheduleId;
using ninfer::ops::detail::Q4ScheduleId;

using S = Q4LinearSwiGluScheduleId;

int failures = 0;

template <class Fn>
void expect_invalid(const char* label, Fn&& fn) {
    try {
        fn();
        std::cerr << label << ": expected invalid_argument\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
}

S expected_schedule(std::int32_t cols) {
    if (cols == 1) { return S::GemvPair; }
    if (cols <= 128 || (cols >= 257 && cols <= 384) || (cols >= 513 && cols <= 640)) {
        return S::Materialized;
    }
    return S::MmaSplitHalfPairR32C128;
}

Q4ScheduleId expected_materialized_schedule(std::int32_t cols) {
    if (cols <= 4) { return Q4ScheduleId::SimtR8C4; }
    if (cols <= 16) { return Q4ScheduleId::SimtR8C8; }
    return Q4ScheduleId::MmaR64C128;
}

void route_tests() {
    constexpr std::array<std::int32_t, 20> boundaries{
        1,
        2,
        4,
        5,
        16,
        17,
        128,
        129,
        256,
        257,
        384,
        385,
        512,
        513,
        640,
        641,
        1024,
        1025,
        ninfer::ops::detail::kQ4LinearSwiGluMaxCols - 1,
        ninfer::ops::detail::kQ4LinearSwiGluMaxCols,
    };
    for (const std::int32_t cols : boundaries) {
        const Q4LinearSwiGluProblem problem{34816, 17408, 5120, 5120, cols};
        if (!ninfer::ops::detail::q4_linear_swiglu_admits(problem)) {
            std::cerr << "rejected admitted Q4 LinearSwiGLU problem C=" << cols << '\n';
            ++failures;
            continue;
        }
        const Q4LinearSwiGluPlan plan = ninfer::ops::detail::q4_linear_swiglu_resolve_plan(problem);
        if (plan.schedule != expected_schedule(cols)) {
            std::cerr << "wrong Q4 LinearSwiGLU route C=" << cols << '\n';
            ++failures;
        }
        if (plan.performance_qualified !=
            (cols <= ninfer::ops::detail::kQ4LinearSwiGluQualifiedCols)) {
            std::cerr << "wrong Q4 LinearSwiGLU qualification C=" << cols << '\n';
            ++failures;
        }
        const bool materialized = plan.schedule == S::Materialized;
        if (plan.materialized_projection.has_value() != materialized ||
            (plan.workspace_bytes != 0) != materialized) {
            std::cerr << "wrong Q4 LinearSwiGLU subplan/workspace C=" << cols << '\n';
            ++failures;
        }
        if (materialized &&
            plan.materialized_projection->schedule != expected_materialized_schedule(cols)) {
            std::cerr << "wrong materialized Q4 schedule C=" << cols << '\n';
            ++failures;
        }
    }

    expect_invalid("C=0", [] {
        (void)ninfer::ops::detail::q4_linear_swiglu_resolve_plan({34816, 17408, 5120, 5120, 0});
    });
    expect_invalid("C=max+1", [] {
        (void)ninfer::ops::detail::q4_linear_swiglu_resolve_plan(
            {34816, 17408, 5120, 5120, ninfer::ops::detail::kQ4LinearSwiGluMaxCols + 1});
    });
    expect_invalid("unsupported output rows", [] {
        (void)ninfer::ops::detail::q4_linear_swiglu_resolve_plan({34816, 17407, 5120, 5120, 1});
    });
    expect_invalid("unsupported K", [] {
        (void)ninfer::ops::detail::q4_linear_swiglu_resolve_plan({34816, 17408, 6144, 6144, 1});
    });
}

void workspace_tests() {
    struct Case {
        std::int32_t capacity;
        std::size_t bytes;
    };

    constexpr std::array<Case, 16> cases{{
        {1, 0},
        {2, 139'264},
        {128, 8'912'896},
        {129, 8'912'896},
        {256, 8'912'896},
        {257, 17'895'424},
        {384, 26'738'688},
        {385, 26'738'688},
        {512, 26'738'688},
        {513, 35'721'216},
        {640, 44'564'480},
        {641, 44'564'480},
        {1024, 44'564'480},
        {1025, 44'564'480},
        {2048, 44'564'480},
        {ninfer::ops::detail::kQ4LinearSwiGluMaxCols, 44'564'480},
    }};
    for (const Case test : cases) {
        const std::size_t actual = ninfer::ops::linear_swiglu_workspace_bytes(34816, test.capacity);
        if (actual != test.bytes) {
            std::cerr << "workspace C=" << test.capacity << ": expected " << test.bytes << ", got "
                      << actual << '\n';
            ++failures;
        }
    }
    expect_invalid("workspace unsupported rows",
                   [] { (void)ninfer::ops::linear_swiglu_workspace_bytes(34814, 1); });
    expect_invalid("workspace C0",
                   [] { (void)ninfer::ops::linear_swiglu_workspace_bytes(34816, 0); });
}

} // namespace

int main() {
    route_tests();
    workspace_tests();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " Q4 LinearSwiGLU plan\n";
    return failures == 0 ? 0 : 1;
}
