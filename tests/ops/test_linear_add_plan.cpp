#include "ninfer/ops/linear_add.h"
#include "ops/linear_add/q5/q5_linear_add_plan.h"
#include "ops/linear_add/w8/w8_linear_add_plan.h"

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
    constexpr std::array<std::int32_t, 10> boundaries{
        1, 2, 24, 25, 128, 129, 512, 1024, 1025, 2048,
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
        expect("C2048", ninfer::ops::linear_add_workspace_bytes(5120, k, 2048), 245'760);
    }
    expect_invalid("workspace unsupported shape",
                   [] { (void)ninfer::ops::linear_add_workspace_bytes(5120, 5120, 1); });
    expect_invalid("workspace C0",
                   [] { (void)ninfer::ops::linear_add_workspace_bytes(5120, 6144, 0); });

    for (const std::int32_t cols : {1, 52, 53, 640, 641, 1024, 1025, 2048}) {
        expect("W8 workspace", ninfer::ops::linear_add_workspace_bytes(2048, 4096, cols), 0);
    }
    expect_invalid("W8 workspace C0",
                   [] { (void)ninfer::ops::linear_add_workspace_bytes(2048, 4096, 0); });
}

void w8_route_tests() {
    using ninfer::ops::detail::W8KernelVariant;
    using ninfer::ops::detail::W8LinearAddScheduleId;
    constexpr std::array<std::int32_t, 25> boundaries{
        1,  2,  3,  4,  5,  6,  7,   8,   9,    10,   11,   12, 13,
        14, 15, 16, 17, 52, 53, 640, 641, 1024, 1025, 2048, 0,
    };
    for (const std::int32_t cols : boundaries) {
        const ninfer::ops::detail::W8Problem problem{2048, 4096, 4096, cols};
        const bool admitted = cols >= 1;
        if (ninfer::ops::detail::w8_linear_add_admits(problem) != admitted) {
            std::cerr << "wrong W8 LinearAdd admission C=" << cols << '\n';
            ++failures;
            continue;
        }
        if (!admitted) { continue; }
        const auto plan = ninfer::ops::detail::w8_linear_add_resolve_plan(problem);
        const W8LinearAddScheduleId expected = cols <= 8    ? W8LinearAddScheduleId::SimtR8C4
                                               : cols <= 16 ? W8LinearAddScheduleId::SplitKMmaExactT
                                               : cols <= 52 ? W8LinearAddScheduleId::SimtR8C4
                                               : cols <= 640 ? W8LinearAddScheduleId::MmaR32C128
                                                             : W8LinearAddScheduleId::MmaR64C128;
        const W8KernelVariant expected_variant =
            expected == W8LinearAddScheduleId::SplitKMmaExactT
                ? W8KernelVariant::None
                : (cols % (expected == W8LinearAddScheduleId::SimtR8C4 ? 4 : 128) == 0
                       ? W8KernelVariant::Full
                       : W8KernelVariant::Predicated);
        if (plan.schedule != expected || plan.variant != expected_variant) {
            std::cerr << "wrong W8 LinearAdd route C=" << cols << '\n';
            ++failures;
        }
    }
    expect_invalid("W8 unsupported rows", [] {
        (void)ninfer::ops::detail::w8_linear_add_resolve_plan({2049, 4096, 4096, 1});
    });
    expect_invalid("W8 unsupported K", [] {
        (void)ninfer::ops::detail::w8_linear_add_resolve_plan({2048, 4608, 4608, 1});
    });
}

} // namespace

int main() {
    route_tests();
    w8_route_tests();
    workspace_tests();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearAdd plans\n";
    return failures == 0 ? 0 : 1;
}
