#include "ninfer/ops/linear_add.h"
#include "ops/linear_add/q5/q5_linear_add_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

using ninfer::ops::detail::Q5LinearAddPlan;
using ninfer::ops::detail::Q5LinearAddProblem;
using ninfer::ops::detail::Q5LinearAddScheduleId;

using S = Q5LinearAddScheduleId;

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
    if (cols == 1) { return S::GemvResidual; }
    if (cols <= 24) { return S::Materialized; }
    if (cols <= 128) { return S::MmaResidualR64C64; }
    return S::MmaResidualR64C128;
}

void route_tests() {
    constexpr std::array<std::int32_t, 12> boundaries{
        1,
        2,
        24,
        25,
        128,
        129,
        512,
        1024,
        1025,
        2048,
        ninfer::ops::detail::kQ5LinearAddMaxCols - 1,
        ninfer::ops::detail::kQ5LinearAddMaxCols,
    };
    for (const std::int32_t k : {6144, 17408}) {
        for (const std::int32_t cols : boundaries) {
            const Q5LinearAddProblem problem{5120, k, k, cols};
            if (!ninfer::ops::detail::q5_linear_add_admits(problem)) {
                std::cerr << "rejected admitted Q5 LinearAdd problem K=" << k << " C=" << cols
                          << '\n';
                ++failures;
                continue;
            }
            const Q5LinearAddPlan plan = ninfer::ops::detail::q5_linear_add_resolve_plan(problem);
            if (plan.schedule != expected_schedule(cols)) {
                std::cerr << "wrong Q5 LinearAdd route K=" << k << " C=" << cols << '\n';
                ++failures;
            }
            if (plan.performance_qualified !=
                (cols <= ninfer::ops::detail::kQ5LinearAddQualifiedCols)) {
                std::cerr << "wrong Q5 LinearAdd qualification K=" << k << " C=" << cols << '\n';
                ++failures;
            }
            const bool materialized = plan.schedule == S::Materialized;
            if (plan.materialized_projection.has_value() != materialized ||
                (plan.workspace_bytes != 0) != materialized) {
                std::cerr << "wrong Q5 LinearAdd subplan/workspace K=" << k << " C=" << cols
                          << '\n';
                ++failures;
            }
        }
    }

    expect_invalid("C=0", [] {
        (void)ninfer::ops::detail::q5_linear_add_resolve_plan({5120, 6144, 6144, 0});
    });
    expect_invalid("C=max+1", [] {
        (void)ninfer::ops::detail::q5_linear_add_resolve_plan(
            {5120, 6144, 6144, ninfer::ops::detail::kQ5LinearAddMaxCols + 1});
    });
    expect_invalid("unsupported rows", [] {
        (void)ninfer::ops::detail::q5_linear_add_resolve_plan({4096, 6144, 6144, 1});
    });
    expect_invalid("unsupported K", [] {
        (void)ninfer::ops::detail::q5_linear_add_resolve_plan({5120, 5120, 5120, 1});
    });
}

void workspace_tests() {
    const auto expect = [](const char* label, std::size_t actual, std::size_t expected) {
        if (actual != expected) {
            std::cerr << label << ": expected " << expected << ", got " << actual << '\n';
            ++failures;
        }
    };
    for (const std::int32_t k : {6144, 17408}) {
        expect("C1", ninfer::ops::linear_add_workspace_bytes(5120, k, 1), 0);
        expect("C2", ninfer::ops::linear_add_workspace_bytes(5120, k, 2), 20'480);
        expect("C24", ninfer::ops::linear_add_workspace_bytes(5120, k, 24), 245'760);
        expect("C25", ninfer::ops::linear_add_workspace_bytes(5120, k, 25), 245'760);
        expect("C1024", ninfer::ops::linear_add_workspace_bytes(5120, k, 1024), 245'760);
        expect("Cmax",
               ninfer::ops::linear_add_workspace_bytes(5120, k,
                                                       ninfer::ops::detail::kQ5LinearAddMaxCols),
               245'760);
    }
    expect_invalid("workspace unsupported shape",
                   [] { (void)ninfer::ops::linear_add_workspace_bytes(5120, 5120, 1); });
    expect_invalid("workspace C0",
                   [] { (void)ninfer::ops::linear_add_workspace_bytes(5120, 6144, 0); });
}

} // namespace

int main() {
    route_tests();
    workspace_tests();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " Q5 LinearAdd plan\n";
    return failures == 0 ? 0 : 1;
}
