#include "ops/linear/q5/q5_rowsplit_plan.h"

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

using ninfer::ops::detail::Q5KernelVariant;
using ninfer::ops::detail::Q5Plan;
using ninfer::ops::detail::Q5Problem;
using ninfer::ops::detail::Q5ScheduleId;

namespace {

using S = Q5ScheduleId;
using V = Q5KernelVariant;

int failures = 0;

void fail(const std::string& label, const std::string& message) {
    std::cerr << "FAIL " << label << ": " << message << '\n';
    ++failures;
}

std::string plan_name(Q5Plan plan) {
    return std::string(ninfer::ops::detail::q5_schedule_name(plan.schedule)) + "." +
           ninfer::ops::detail::q5_kernel_variant_name(plan.variant);
}

V expected_variant(S schedule, const Q5Problem& problem) {
    if (!ninfer::ops::detail::q5_schedule_uses_mma(schedule)) { return V::None; }
    const int tile_cols = schedule == S::MmaR64C64 ? 64 : 128;
    return problem.rows % 64 == 0 && problem.cols % tile_cols == 0 &&
                   problem.k == problem.padded_k && problem.k % 64 == 0
               ? V::Full
               : V::Predicated;
}

S expected_schedule(const Q5Problem& problem) {
    if (problem.rows == 1024 && problem.k == 5120 && problem.padded_k == 5120) {
        return problem.cols <= 4 ? S::SimtR8C4 : S::SimtR8C8;
    }
    if (problem.rows == 6144 && problem.k == 5120 && problem.padded_k == 5120) {
        if (problem.cols == 1) { return S::GemvR16S2X; }
        if (problem.cols <= 6) { return S::SimtSplit4Exact; }
        if (problem.cols <= 24) { return S::SimtR8C8; }
        if (problem.cols <= 64) { return S::MmaR64C64; }
        return S::MmaR64C128;
    }
    if (problem.rows == 5120 && problem.k == 6144 && problem.padded_k == 6144) {
        return problem.cols <= 6 ? S::SimtSplit2Exact : S::SimtR8C8;
    }
    if (problem.rows == 5120 && problem.k == 17408 && problem.padded_k == 17408) {
        return problem.cols <= 6 ? S::SimtSplit2Exact : S::SimtR8C8;
    }
    if (problem.rows == 1152 && problem.k == 1152 && problem.padded_k == 1152) {
        if (problem.cols == 4) { return S::SimtR8C4; }
        if (problem.cols <= 56) { return S::SimtR8C8; }
        return S::MmaR64C128;
    }
    if (problem.rows == 1152 && problem.k == 4304 && problem.padded_k == 4352) {
        if (problem.cols == 4) { return S::SimtR8C4; }
        if (problem.cols <= 84) { return S::SimtR8C8; }
        return S::MmaR64C128;
    }
    throw std::logic_error("Q5 test oracle received an unregistered problem");
}

void expect_plan(const std::string& label, const Q5Problem& problem) {
    if (!ninfer::ops::detail::q5_rowsplit_admits(problem)) {
        fail(label, "production admission rejected the problem");
        return;
    }
    const S schedule = expected_schedule(problem);
    const Q5Plan expected{schedule, expected_variant(schedule, problem)};
    try {
        const Q5Plan actual = ninfer::ops::detail::q5_rowsplit_resolve_plan(problem);
        if (actual.schedule != expected.schedule || actual.variant != expected.variant) {
            fail(label, "expected " + plan_name(expected) + ", got " + plan_name(actual));
        }
        if (!ninfer::ops::detail::q5_candidate_is_legal(actual.schedule, actual.variant, problem)) {
            fail(label, "resolver returned a physically illegal plan");
        }
    } catch (const std::exception& error) {
        fail(label, std::string("resolver threw: ") + error.what());
    }
}

void admission_scans() {
    for (std::int32_t cols = 0; cols <= 17; ++cols) {
        const Q5Problem problem{1024, 5120, 5120, cols};
        const bool admitted = cols >= 1 && cols <= 16;
        if (ninfer::ops::detail::q5_rowsplit_admits(problem) != admitted) {
            fail("attention value admission",
                 admitted ? "rejected admitted cols" : "accepted rejected cols");
        }
        if (admitted) { expect_plan("attention value route", problem); }
    }

    for (const std::int32_t k : {6144, 17408}) {
        for (std::int32_t cols = 0; cols <= 25; ++cols) {
            const Q5Problem problem{5120, k, k, cols};
            const bool admitted = cols >= 2 && cols <= 24;
            if (ninfer::ops::detail::q5_rowsplit_admits(problem) != admitted) {
                fail("residual materialization admission",
                     admitted ? "rejected admitted cols" : "accepted rejected cols");
            }
            if (admitted) { expect_plan("residual materialization route", problem); }
        }
    }

    for (const Q5Problem base : {Q5Problem{1152, 1152, 1152, 0}, Q5Problem{1152, 4304, 4352, 0}}) {
        for (std::int32_t cols = 0; cols <= 131076; ++cols) {
            Q5Problem problem   = base;
            problem.cols        = cols;
            const bool admitted = cols >= 4 && cols <= 131072 && cols % 4 == 0;
            if (ninfer::ops::detail::q5_rowsplit_admits(problem) != admitted) {
                fail("vision admission",
                     admitted ? "rejected admitted cols" : "accepted rejected cols");
            }
            if (admitted) { expect_plan("vision route", problem); }
        }
    }
}

void route_boundaries() {
    constexpr std::array<Q5Problem, 28> cases{{
        {1024, 5120, 5120, 1},         {1024, 5120, 5120, 4},  {1024, 5120, 5120, 5},
        {1024, 5120, 5120, 16},        {6144, 5120, 5120, 1},  {6144, 5120, 5120, 2},
        {6144, 5120, 5120, 6},         {6144, 5120, 5120, 7},  {6144, 5120, 5120, 24},
        {6144, 5120, 5120, 25},        {6144, 5120, 5120, 64}, {6144, 5120, 5120, 65},
        {6144, 5120, 5120, 8'388'480}, {5120, 6144, 6144, 2},  {5120, 6144, 6144, 6},
        {5120, 6144, 6144, 7},         {5120, 6144, 6144, 24}, {5120, 17408, 17408, 2},
        {5120, 17408, 17408, 24},      {1152, 1152, 1152, 4},  {1152, 1152, 1152, 8},
        {1152, 1152, 1152, 56},        {1152, 1152, 1152, 60}, {1152, 1152, 1152, 131072},
        {1152, 4304, 4352, 4},         {1152, 4304, 4352, 84}, {1152, 4304, 4352, 88},
        {1152, 4304, 4352, 131072},
    }};
    for (const Q5Problem& problem : cases) { expect_plan("route boundary", problem); }
}

void rejection_contract() {
    constexpr std::array<Q5Problem, 9> rejected{{
        {1024, 5120, 5120, 17},
        {7168, 5120, 5120, 1},
        {5120, 6144, 6144, 1},
        {5120, 17408, 17408, 25},
        {1152, 1152, 1152, 5},
        {1152, 4304, 4352, 86},
        {1152, 4304, 4304, 88},
        {6144, 5120, 5120, 8'388'481},
        {6144, 2048, 2048, 1},
    }};
    for (const Q5Problem& problem : rejected) {
        if (ninfer::ops::detail::q5_rowsplit_admits(problem)) {
            fail("rejection", "production admission accepted an unregistered problem");
        }
        try {
            const Q5Plan plan = ninfer::ops::detail::q5_rowsplit_resolve_plan(problem);
            fail("rejection", "resolver returned " + plan_name(plan));
        } catch (const std::invalid_argument&) {
        } catch (const std::exception& error) {
            fail("rejection", std::string("resolver threw wrong exception: ") + error.what());
        }
    }
}

} // namespace

int main() {
    admission_scans();
    route_boundaries();
    rejection_contract();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " Q5 Linear production plan\n";
    return failures == 0 ? 0 : 1;
}
