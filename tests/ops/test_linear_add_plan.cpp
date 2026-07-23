#include "ninfer/ops/linear_add.h"
#include "ops/linear_add/q5/q5_linear_add_plan.h"
#include "ops/linear_add/w8/w8_linear_add_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
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

    for (const std::int32_t k : {4096, 6144}) {
        for (const std::int32_t cols : {1, 52, 53, 640, 641, 1024, 1025, 2048}) {
            expect("W8 workspace", ninfer::ops::linear_add_workspace_bytes(2048, k, cols), 0);
        }
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

    struct DFlashRoute {
        std::int32_t first;
        std::int32_t last;
        W8LinearAddScheduleId schedule;
        std::int32_t tile_rows;
        std::int32_t tile_cols;
    };

    constexpr std::array<DFlashRoute, 36> dflash_routes{{
        {1, 1, W8LinearAddScheduleId::DecodeR16, 0, 0},
        {2, 32, W8LinearAddScheduleId::SplitKMmaExactT, 0, 0},
        {33, 65, W8LinearAddScheduleId::SplitKMma32PlusTail, 0, 0},
        {66, 95, W8LinearAddScheduleId::MmaR32C64, 32, 64},
        {96, 96, W8LinearAddScheduleId::MmaR32C96, 32, 96},
        {97, 128, W8LinearAddScheduleId::MmaR32C64, 32, 64},
        {129, 191, W8LinearAddScheduleId::MmaR32C128, 32, 128},
        {192, 192, W8LinearAddScheduleId::MmaR32C96, 32, 96},
        {193, 256, W8LinearAddScheduleId::MmaR32C128, 32, 128},
        {257, 384, W8LinearAddScheduleId::MmaR32C64, 32, 64},
        {385, 399, W8LinearAddScheduleId::MmaR32C96, 32, 96},
        {400, 400, W8LinearAddScheduleId::MmaR32C80, 32, 80},
        {401, 447, W8LinearAddScheduleId::MmaR32C96, 32, 96},
        {448, 448, W8LinearAddScheduleId::MmaR32C64, 32, 64},
        {449, 480, W8LinearAddScheduleId::MmaR32C96, 32, 96},
        {481, 640, W8LinearAddScheduleId::MmaR32C128, 32, 128},
        {641, 672, W8LinearAddScheduleId::MmaR48C96, 48, 96},
        {673, 704, W8LinearAddScheduleId::MmaR48C64, 48, 64},
        {705, 784, W8LinearAddScheduleId::MmaR48C112, 48, 112},
        {785, 896, W8LinearAddScheduleId::MmaR48C128, 48, 128},
        {897, 960, W8LinearAddScheduleId::MmaR64C96, 64, 96},
        {961, 1023, W8LinearAddScheduleId::MmaR64C112, 64, 112},
        {1024, 1024, W8LinearAddScheduleId::MmaR64C128, 64, 128},
        {1025, 1120, W8LinearAddScheduleId::MmaR64C112, 64, 112},
        {1121, 1280, W8LinearAddScheduleId::MmaR64C128, 64, 128},
        {1281, 1344, W8LinearAddScheduleId::MmaR128C64, 128, 64},
        {1345, 1408, W8LinearAddScheduleId::MmaR48C128, 48, 128},
        {1409, 1680, W8LinearAddScheduleId::MmaR128C80, 128, 80},
        {1681, 1791, W8LinearAddScheduleId::MmaR48C128, 48, 128},
        {1792, 1792, W8LinearAddScheduleId::MmaR64C128, 64, 128},
        {1793, 1919, W8LinearAddScheduleId::MmaR48C128, 48, 128},
        {1920, 1920, W8LinearAddScheduleId::MmaR64C128, 64, 128},
        {1921, 2016, W8LinearAddScheduleId::MmaR64C96, 64, 96},
        {2017, 2047, W8LinearAddScheduleId::MmaR64C112, 64, 112},
        {2048, 2048, W8LinearAddScheduleId::MmaR64C128, 64, 128},
        {2049, std::numeric_limits<std::int32_t>::max(), W8LinearAddScheduleId::MmaR64C128, 64,
         128},
    }};
    for (const DFlashRoute& route : dflash_routes) {
        for (const std::int32_t cols : {route.first, route.last}) {
            const auto plan =
                ninfer::ops::detail::w8_linear_add_resolve_plan({2048, 6144, 6144, cols});
            const W8KernelVariant expected_variant =
                route.schedule == W8LinearAddScheduleId::DecodeR16 ||
                        route.schedule == W8LinearAddScheduleId::SplitKMmaExactT ||
                        route.schedule == W8LinearAddScheduleId::SplitKMma32PlusTail
                    ? W8KernelVariant::None
                    : (2048 % route.tile_rows == 0 && cols % route.tile_cols == 0
                           ? W8KernelVariant::Full
                           : W8KernelVariant::Predicated);
            if (plan.schedule != route.schedule || plan.variant != expected_variant) {
                std::cerr << "wrong W8 [2048,6144] LinearAdd route C=" << cols << '\n';
                ++failures;
            }
        }
    }
    if (!ninfer::ops::detail::w8_linear_add_admits({2048, 6144, 6144, 1})) {
        std::cerr << "W8 [2048,6144] LinearAdd admission missing\n";
        ++failures;
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
