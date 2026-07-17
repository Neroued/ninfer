#include "ops/linear/q4/q4_rowsplit_plan.h"

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

using ninfer::ops::detail::Q4KernelVariant;
using ninfer::ops::detail::Q4Plan;
using ninfer::ops::detail::Q4Problem;
using ninfer::ops::detail::Q4ScheduleId;

namespace {

using S = Q4ScheduleId;
using V = Q4KernelVariant;

int failures = 0;

std::string problem_name(const Q4Problem& problem) {
    return "R=" + std::to_string(problem.rows) + " K=" + std::to_string(problem.k) +
           " Kpad=" + std::to_string(problem.padded_k) + " C=" + std::to_string(problem.cols);
}

void fail(const std::string& label, const std::string& message) {
    std::cerr << "FAIL " << label << ": " << message << '\n';
    ++failures;
}

bool same_plan(Q4Plan lhs, Q4Plan rhs) {
    return lhs.schedule == rhs.schedule && lhs.variant == rhs.variant;
}

std::string plan_name(Q4Plan plan) {
    return std::string(ninfer::ops::detail::q4_schedule_name(plan.schedule)) + "." +
           ninfer::ops::detail::q4_kernel_variant_name(plan.variant);
}

Q4KernelVariant expected_tiled_variant(Q4ScheduleId schedule, const Q4Problem& problem) {
    switch (schedule) {
    case S::SimtR8C4:
        return (problem.cols % 4) == 0 && (problem.rows % 8) == 0 && ((problem.k / 64) % 16) == 0
                   ? V::Full
                   : V::Predicated;
    case S::SimtR8C8:
        return (problem.cols % 8) == 0 && (problem.rows % 8) == 0 && ((problem.k / 64) % 16) == 0
                   ? V::Full
                   : V::Predicated;
    case S::MmaR64C64:
        return (problem.cols % 64) == 0 && (problem.rows % 64) == 0 ? V::Full : V::Predicated;
    case S::MmaR64C128:
        return (problem.cols % 128) == 0 && (problem.rows % 64) == 0 ? V::Full : V::Predicated;
    case S::GemvR4W1Direct:
    case S::GemvR1W8Direct:
        return V::None;
    }
    throw std::logic_error("unexpected Q4 schedule in test oracle");
}

Q4Plan expected_production_plan(const Q4Problem& problem) {
    S schedule;
    if (problem.rows == 1024 && problem.k == 5120) {
        schedule = problem.cols == 1
                       ? S::GemvR1W8Direct
                       : (problem.cols <= 15 ? S::SimtR8C4
                                             : (problem.cols == 16 ? S::SimtR8C8 : S::MmaR64C128));
    } else if (problem.rows == 4096 && problem.k == 5120) {
        schedule = problem.cols == 1
                       ? S::GemvR1W8Direct
                       : (problem.cols <= 4 ? S::SimtR8C4
                                            : (problem.cols <= 16 ? S::SimtR8C8 : S::MmaR64C128));
    } else if (problem.rows == 6144 && problem.k == 5120) {
        schedule = problem.cols == 1
                       ? S::GemvR1W8Direct
                       : (problem.cols <= 7 ? S::SimtR8C4
                                            : (problem.cols <= 16 ? S::SimtR8C8 : S::MmaR64C128));
    } else if (problem.rows == 7168 && problem.k == 5120) {
        if (problem.cols == 1) {
            schedule = S::GemvR1W8Direct;
        } else if (problem.cols <= 7 || (problem.cols >= 9 && problem.cols <= 15)) {
            schedule = S::SimtR8C4;
        } else if (problem.cols == 8 || problem.cols == 16) {
            schedule = S::SimtR8C8;
        } else {
            schedule = S::MmaR64C128;
        }
    } else if (problem.rows == 34816 && problem.k == 5120) {
        schedule = problem.cols == 1
                       ? S::GemvR1W8Direct
                       : (problem.cols <= 4 ? S::SimtR8C4
                                            : (problem.cols <= 16 ? S::SimtR8C8 : S::MmaR64C128));
    } else if (problem.rows == 131072 && (problem.k == 5120 || problem.k == 2048)) {
        schedule = problem.cols == 1 ? S::GemvR4W1Direct : S::MmaR64C128;
    } else if (problem.rows == 3456 && problem.k == 1152) {
        schedule =
            problem.cols <= 36 ? S::SimtR8C4 : (problem.cols <= 320 ? S::MmaR64C64 : S::MmaR64C128);
    } else if (problem.rows == 4304 && problem.k == 1152) {
        if (problem.cols == 4 || problem.cols == 12) {
            schedule = S::SimtR8C4;
        } else if (problem.cols == 8 || problem.cols <= 24) {
            schedule = S::SimtR8C8;
        } else {
            schedule = problem.cols <= 320 ? S::MmaR64C64 : S::MmaR64C128;
        }
    } else {
        throw std::logic_error("test oracle received an unregistered problem");
    }
    return {schedule, expected_tiled_variant(schedule, problem)};
}

void expect_plan(const std::string& label, const Q4Problem& problem, Q4Plan expected) {
    if (!ninfer::ops::detail::q4_rowsplit_admits(problem)) {
        fail(label, "production admission rejected " + problem_name(problem));
        return;
    }

    try {
        const Q4Plan actual = ninfer::ops::detail::q4_rowsplit_resolve_plan(problem);
        if (!same_plan(actual, expected)) {
            fail(label, "expected " + plan_name(expected) + ", got " + plan_name(actual));
        }
        if (!ninfer::ops::detail::q4_candidate_is_legal(actual.schedule, actual.variant, problem)) {
            fail(label, "resolver returned a physically illegal plan");
        }
    } catch (const std::exception& error) {
        fail(label, std::string("resolver threw: ") + error.what());
    }
}

constexpr std::array<S, 6> kSchedules{{
    S::GemvR4W1Direct,
    S::GemvR1W8Direct,
    S::SimtR8C4,
    S::SimtR8C8,
    S::MmaR64C64,
    S::MmaR64C128,
}};

constexpr std::array<V, 3> kVariants{{V::None, V::Full, V::Predicated}};

bool has_legal_candidate(const Q4Problem& problem) {
    for (S schedule : kSchedules) {
        for (V variant : kVariants) {
            if (ninfer::ops::detail::q4_candidate_is_legal(schedule, variant, problem)) {
                return true;
            }
        }
    }
    return false;
}

void expect_rejected(const std::string& label, const Q4Problem& problem, bool expect_capability) {
    if (ninfer::ops::detail::q4_rowsplit_admits(problem)) {
        fail(label, "production admission accepted " + problem_name(problem));
    }
    if (has_legal_candidate(problem) != expect_capability) {
        fail(label, expect_capability ? "expected at least one physically legal candidate"
                                      : "unexpected physically legal candidate");
    }

    try {
        const Q4Plan plan = ninfer::ops::detail::q4_rowsplit_resolve_plan(problem);
        fail(label, "resolver returned " + plan_name(plan));
    } catch (const std::invalid_argument&) {
    } catch (const std::exception& error) {
        fail(label, std::string("resolver threw wrong exception: ") + error.what());
    }
}

void semantic_length_admission() {
    constexpr std::array<Q4Problem, 7> text_shapes{{
        {1024, 5120, 5120, 1},
        {4096, 5120, 5120, 1},
        {6144, 5120, 5120, 1},
        {7168, 5120, 5120, 1},
        {34816, 5120, 5120, 1},
        {131072, 5120, 5120, 1},
        {131072, 2048, 2048, 1},
    }};
    constexpr std::array<std::int32_t, 8> positive_t{1, 2, 5, 16, 17, 1024, 1025, 2048};

    for (Q4Problem problem : text_shapes) {
        problem.cols = 0;
        if (ninfer::ops::detail::q4_rowsplit_admits(problem)) {
            fail("T=0 rejection", "accepted " + problem_name(problem));
        }
        for (const std::int32_t cols : positive_t) {
            problem.cols = cols;
            expect_plan("arbitrary positive T", problem, expected_production_plan(problem));
        }
    }

    constexpr std::array<Q4Problem, 8> valid_vision{{
        {3456, 1152, 1152, 4},
        {3456, 1152, 1152, 1024},
        {3456, 1152, 1152, 131072},
        {4304, 1152, 1152, 4},
        {4304, 1152, 1152, 8},
        {4304, 1152, 1152, 1024},
        {4304, 1152, 1152, 1028},
        {4304, 1152, 1152, 131072},
    }};
    for (const Q4Problem& problem : valid_vision) {
        expect_plan("valid Vision P", problem, expected_production_plan(problem));
    }
}

struct RouteBoundaryCase {
    const char* label;
    Q4Problem problem;
    Q4Plan expected;
};

void route_boundaries_and_seams() {
    constexpr std::array<RouteBoundaryCase, 46> cases{{
        {"1024 gemv end", {1024, 5120, 5120, 1}, {S::GemvR1W8Direct, V::None}},
        {"1024 gemv/simt seam", {1024, 5120, 5120, 2}, {S::SimtR8C4, V::Predicated}},
        {"1024 c4 end", {1024, 5120, 5120, 15}, {S::SimtR8C4, V::Predicated}},
        {"1024 c4/c8 seam", {1024, 5120, 5120, 16}, {S::SimtR8C8, V::Full}},
        {"1024 general route", {1024, 5120, 5120, 17}, {S::MmaR64C128, V::Predicated}},
        {"1024 beyond default", {1024, 5120, 5120, 1025}, {S::MmaR64C128, V::Predicated}},

        {"4096 gemv end", {4096, 5120, 5120, 1}, {S::GemvR1W8Direct, V::None}},
        {"4096 gemv/c4 seam", {4096, 5120, 5120, 2}, {S::SimtR8C4, V::Predicated}},
        {"4096 c4 end", {4096, 5120, 5120, 4}, {S::SimtR8C4, V::Full}},
        {"4096 c4/c8 seam", {4096, 5120, 5120, 5}, {S::SimtR8C8, V::Predicated}},
        {"4096 c8 end", {4096, 5120, 5120, 16}, {S::SimtR8C8, V::Full}},

        {"6144 gemv end", {6144, 5120, 5120, 1}, {S::GemvR1W8Direct, V::None}},
        {"6144 gemv/c4 seam", {6144, 5120, 5120, 2}, {S::SimtR8C4, V::Predicated}},
        {"6144 c4 end", {6144, 5120, 5120, 7}, {S::SimtR8C4, V::Predicated}},
        {"6144 c4/c8 seam", {6144, 5120, 5120, 8}, {S::SimtR8C8, V::Full}},
        {"6144 c8 end", {6144, 5120, 5120, 16}, {S::SimtR8C8, V::Full}},

        {"7168 gemv end", {7168, 5120, 5120, 1}, {S::GemvR1W8Direct, V::None}},
        {"7168 gemv/c4 seam", {7168, 5120, 5120, 2}, {S::SimtR8C4, V::Predicated}},
        {"7168 c4 end", {7168, 5120, 5120, 7}, {S::SimtR8C4, V::Predicated}},
        {"7168 c8 singleton", {7168, 5120, 5120, 8}, {S::SimtR8C8, V::Full}},
        {"7168 second c4 begin", {7168, 5120, 5120, 9}, {S::SimtR8C4, V::Predicated}},
        {"7168 second c4 end", {7168, 5120, 5120, 15}, {S::SimtR8C4, V::Predicated}},
        {"7168 c8 end", {7168, 5120, 5120, 16}, {S::SimtR8C8, V::Full}},

        {"34816 c4 begin", {34816, 5120, 5120, 2}, {S::SimtR8C4, V::Predicated}},
        {"34816 c4 end", {34816, 5120, 5120, 4}, {S::SimtR8C4, V::Full}},
        {"34816 c4/c8 seam", {34816, 5120, 5120, 5}, {S::SimtR8C8, V::Predicated}},
        {"34816 c8 end", {34816, 5120, 5120, 16}, {S::SimtR8C8, V::Full}},
        {"34816 T1", {34816, 5120, 5120, 1}, {S::GemvR1W8Direct, V::None}},

        {"131072 gemv singleton", {131072, 5120, 5120, 1}, {S::GemvR4W1Direct, V::None}},
        {"131072 general route", {131072, 5120, 5120, 2}, {S::MmaR64C128, V::Predicated}},
        {"131072 K2048 gemv singleton", {131072, 2048, 2048, 1}, {S::GemvR4W1Direct, V::None}},

        {"3456 c4 begin", {3456, 1152, 1152, 4}, {S::SimtR8C4, V::Predicated}},
        {"3456 c4 end", {3456, 1152, 1152, 36}, {S::SimtR8C4, V::Predicated}},
        {"3456 c4/c64 seam", {3456, 1152, 1152, 40}, {S::MmaR64C64, V::Predicated}},
        {"3456 c64 end", {3456, 1152, 1152, 320}, {S::MmaR64C64, V::Full}},
        {"3456 c64/c128 seam", {3456, 1152, 1152, 324}, {S::MmaR64C128, V::Predicated}},
        {"3456 c128 end", {3456, 1152, 1152, 131072}, {S::MmaR64C128, V::Full}},

        {"4304 c4 singleton", {4304, 1152, 1152, 4}, {S::SimtR8C4, V::Predicated}},
        {"4304 c8 singleton", {4304, 1152, 1152, 8}, {S::SimtR8C8, V::Predicated}},
        {"4304 second c4 singleton", {4304, 1152, 1152, 12}, {S::SimtR8C4, V::Predicated}},
        {"4304 c8 begin", {4304, 1152, 1152, 16}, {S::SimtR8C8, V::Predicated}},
        {"4304 c8 end", {4304, 1152, 1152, 24}, {S::SimtR8C8, V::Predicated}},
        {"4304 c8/c64 seam", {4304, 1152, 1152, 28}, {S::MmaR64C64, V::Predicated}},
        {"4304 c64 end", {4304, 1152, 1152, 320}, {S::MmaR64C64, V::Predicated}},
        {"4304 c64/c128 seam", {4304, 1152, 1152, 324}, {S::MmaR64C128, V::Predicated}},
        {"4304 c128 end", {4304, 1152, 1152, 131072}, {S::MmaR64C128, V::Predicated}},
    }};

    for (const RouteBoundaryCase& test : cases) {
        expect_plan(test.label, test.problem, test.expected);
    }
}

void rejection_contract() {
    constexpr std::array<Q4Problem, 6> capability_valid_unadmitted{{
        {128, 128, 128, 4},
        {17408, 5120, 5120, 8},
        {3456, 1152, 1152, 1},
        {3456, 1152, 1152, 5},
        {4304, 1152, 1152, 131073},
        {4304, 1152, 1152, 131076},
    }};
    for (std::size_t i = 0; i < capability_valid_unadmitted.size(); ++i) {
        expect_rejected("capability-valid unadmitted " + std::to_string(i),
                        capability_valid_unadmitted[i], true);
    }

    constexpr std::array<Q4Problem, 5> capability_invalid{{
        {0, 128, 128, 4},
        {127, 128, 128, 4},
        {128, 130, 130, 4},
        {128, 128, 256, 4},
        {128, 128, 128, 0},
    }};
    for (std::size_t i = 0; i < capability_invalid.size(); ++i) {
        expect_rejected("capability-invalid " + std::to_string(i), capability_invalid[i], false);
    }

    constexpr std::array<Q4Problem, 3> future_k2048{{
        {1024, 2048, 2048, 1},
        {4096, 2048, 2048, 8},
        {34816, 2048, 2048, 16},
    }};
    for (std::size_t i = 0; i < future_k2048.size(); ++i) {
        expect_rejected("future K=2048 " + std::to_string(i), future_k2048[i], true);
    }
}

} // namespace

int main() {
    semantic_length_admission();
    route_boundaries_and_seams();
    rejection_contract();

    std::cout << (failures == 0 ? "OK" : "FAIL") << " Q4 Linear production plan\n";
    return failures == 0 ? 0 : 1;
}
