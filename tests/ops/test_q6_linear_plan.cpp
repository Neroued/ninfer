#include "ops/linear/q6/q6_rowsplit_plan.h"

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

using ninfer::ops::detail::Q6KernelVariant;
using ninfer::ops::detail::Q6Plan;
using ninfer::ops::detail::Q6Problem;
using ninfer::ops::detail::Q6ScheduleId;

namespace {

using S = Q6ScheduleId;
using V = Q6KernelVariant;

int failures = 0;

void fail(const std::string& label, const std::string& message) {
    std::cerr << "FAIL " << label << ": " << message << '\n';
    ++failures;
}

bool same_plan(Q6Plan lhs, Q6Plan rhs) {
    return lhs.schedule == rhs.schedule && lhs.variant == rhs.variant;
}

std::string plan_name(Q6Plan plan) {
    return std::string(ninfer::ops::detail::q6_schedule_name(plan.schedule)) + "." +
           ninfer::ops::detail::q6_kernel_variant_name(plan.variant);
}

Q6KernelVariant expected_variant(Q6ScheduleId schedule, const Q6Problem& problem) {
    if (schedule == S::SimtR8C4 || schedule == S::SimtR8C8) { return V::None; }
    const int cols = schedule == S::MmaR64C64 ? 64 : 128;
    return problem.rows % 64 == 0 && problem.cols % cols == 0 && problem.k == problem.padded_k &&
                   problem.k % 64 == 0
               ? V::Full
               : V::Predicated;
}

Q6Plan expected_plan(const Q6Problem& problem) {
    S schedule;
    if (problem.rows == 248320 && problem.k == 5120) {
        schedule = problem.cols <= 6 ? S::SimtR8C4 : S::MmaR64C128;
    } else if (problem.rows == 248320 && problem.k == 2048) {
        schedule =
            problem.cols <= 4 ? S::SimtR8C4 : (problem.cols <= 6 ? S::SimtR8C8 : S::MmaR64C128);
    } else if (problem.rows == 1152 && problem.k == 1536) {
        const int cols = problem.cols;
        if (cols <= 96) {
            schedule = S::SimtR8C4;
        } else if (cols <= 704) {
            schedule = S::MmaR64C64;
        } else if (cols <= 828) {
            schedule = S::MmaR64C128;
        } else if (cols <= 832) {
            schedule = S::MmaR64C64;
        } else if (cols <= 896) {
            schedule = S::MmaR64C128;
        } else if (cols <= 960) {
            schedule = S::MmaR64C64;
        } else if (cols <= 1024) {
            schedule = S::MmaR64C128;
        } else if (cols <= 1088) {
            schedule = S::MmaR64C64;
        } else {
            schedule = S::MmaR64C128;
        }
    } else {
        throw std::logic_error("test oracle received an unregistered Q6 problem");
    }
    return {schedule, expected_variant(schedule, problem)};
}

void expect_plan(const std::string& label, const Q6Problem& problem, Q6Plan expected) {
    if (!ninfer::ops::detail::q6_rowsplit_admits(problem)) {
        fail(label, "production admission rejected the problem");
        return;
    }
    try {
        const Q6Plan actual = ninfer::ops::detail::q6_rowsplit_resolve_plan(problem);
        if (!same_plan(actual, expected)) {
            fail(label, "expected " + plan_name(expected) + ", got " + plan_name(actual));
        }
        if (!ninfer::ops::detail::q6_candidate_is_legal(actual.schedule, actual.variant, problem)) {
            fail(label, "resolver returned a physically illegal plan");
        }
    } catch (const std::exception& error) {
        fail(label, std::string("resolver threw: ") + error.what());
    }
}

void full_support_scan() {
    constexpr std::array<Q6Problem, 2> text_shapes{{
        {248320, 5120, 5120, 1},
        {248320, 2048, 2048, 1},
    }};
    constexpr std::array<std::int32_t, 8> positive_t{1, 2, 5, 6, 7, 1024, 1025, 2048};
    for (Q6Problem problem : text_shapes) {
        problem.cols = 0;
        if (ninfer::ops::detail::q6_rowsplit_admits(problem)) {
            fail("T=0 rejection", "accepted zero columns");
        }
        for (const std::int32_t cols : positive_t) {
            problem.cols = cols;
            expect_plan("arbitrary positive T", problem, expected_plan(problem));
        }
    }

    constexpr std::array<std::int32_t, 5> valid_p{4, 1024, 1028, 2048, 131072};
    for (const std::int32_t patches : valid_p) {
        const Q6Problem problem{1152, 1536, 1536, patches};
        expect_plan("valid Vision P", problem, expected_plan(problem));
    }
}

struct BoundaryCase {
    Q6Problem problem;
    Q6Plan expected;
};

void route_boundaries() {
    constexpr std::array<BoundaryCase, 27> cases{{
        {{248320, 5120, 5120, 1}, {S::SimtR8C4, V::None}},
        {{248320, 5120, 5120, 6}, {S::SimtR8C4, V::None}},
        {{248320, 5120, 5120, 7}, {S::MmaR64C128, V::Predicated}},
        {{248320, 5120, 5120, 1025}, {S::MmaR64C128, V::Predicated}},
        {{248320, 2048, 2048, 1}, {S::SimtR8C4, V::None}},
        {{248320, 2048, 2048, 4}, {S::SimtR8C4, V::None}},
        {{248320, 2048, 2048, 5}, {S::SimtR8C8, V::None}},
        {{248320, 2048, 2048, 6}, {S::SimtR8C8, V::None}},
        {{248320, 2048, 2048, 7}, {S::MmaR64C128, V::Predicated}},
        {{1152, 1536, 1536, 4}, {S::SimtR8C4, V::None}},
        {{1152, 1536, 1536, 96}, {S::SimtR8C4, V::None}},
        {{1152, 1536, 1536, 100}, {S::MmaR64C64, V::Predicated}},
        {{1152, 1536, 1536, 704}, {S::MmaR64C64, V::Full}},
        {{1152, 1536, 1536, 708}, {S::MmaR64C128, V::Predicated}},
        {{1152, 1536, 1536, 768}, {S::MmaR64C128, V::Full}},
        {{1152, 1536, 1536, 828}, {S::MmaR64C128, V::Predicated}},
        {{1152, 1536, 1536, 832}, {S::MmaR64C64, V::Full}},
        {{1152, 1536, 1536, 836}, {S::MmaR64C128, V::Predicated}},
        {{1152, 1536, 1536, 896}, {S::MmaR64C128, V::Full}},
        {{1152, 1536, 1536, 900}, {S::MmaR64C64, V::Predicated}},
        {{1152, 1536, 1536, 960}, {S::MmaR64C64, V::Full}},
        {{1152, 1536, 1536, 964}, {S::MmaR64C128, V::Predicated}},
        {{1152, 1536, 1536, 1024}, {S::MmaR64C128, V::Full}},
        {{1152, 1536, 1536, 1028}, {S::MmaR64C64, V::Predicated}},
        {{1152, 1536, 1536, 1088}, {S::MmaR64C64, V::Full}},
        {{1152, 1536, 1536, 1092}, {S::MmaR64C128, V::Predicated}},
        {{1152, 1536, 1536, 131072}, {S::MmaR64C128, V::Full}},
    }};

    for (const BoundaryCase& test : cases) {
        expect_plan("route boundary", test.problem, test.expected);
    }
}

void rejection_contract() {
    constexpr std::array<Q6Problem, 6> rejected{{
        {65536, 5120, 5120, 1},
        {4096, 5120, 5120, 4},
        {248320, 2048, 2048, 0},
        {1152, 1536, 1536, 1},
        {1152, 1536, 1536, 5},
        {1152, 1536, 1536, 131076},
    }};

    for (const Q6Problem& problem : rejected) {
        if (ninfer::ops::detail::q6_rowsplit_admits(problem)) {
            fail("rejection", "production admission accepted an unregistered problem");
        }
        try {
            const Q6Plan plan = ninfer::ops::detail::q6_rowsplit_resolve_plan(problem);
            fail("rejection", "resolver returned " + plan_name(plan));
        } catch (const std::invalid_argument&) {
        } catch (const std::exception& error) {
            fail("rejection", std::string("resolver threw wrong exception: ") + error.what());
        }
    }
}

} // namespace

int main() {
    full_support_scan();
    route_boundaries();
    rejection_contract();

    std::cout << (failures == 0 ? "OK" : "FAIL") << " Q6 Linear production plan\n";
    return failures == 0 ? 0 : 1;
}
