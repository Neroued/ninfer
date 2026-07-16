#include "ops/linear/w8/w8_rowsplit_plan.h"
#include "ops/linear_pair/w8/w8_pair_plan.h"

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

using ninfer::ops::detail::W8KernelVariant;
using ninfer::ops::detail::W8PairPlan;
using ninfer::ops::detail::W8PairProblem;
using ninfer::ops::detail::W8PairScheduleId;
using ninfer::ops::detail::W8Plan;
using ninfer::ops::detail::W8Problem;
using ninfer::ops::detail::W8ScheduleId;

namespace {

using S = W8ScheduleId;
using V = W8KernelVariant;

int failures = 0;

void fail(const std::string& label, const std::string& message) {
    std::cerr << "FAIL " << label << ": " << message << '\n';
    ++failures;
}

std::string plan_name(W8Plan plan) {
    return std::string(ninfer::ops::detail::w8_schedule_name(plan.schedule)) + "." +
           ninfer::ops::detail::w8_kernel_variant_name(plan.variant);
}

S expected_schedule(const W8Problem& problem) {
    if (problem.rows == 14336 || problem.rows == 34816) {
        if (problem.cols <= 4) { return S::SimtR8C4; }
        if (problem.cols <= 8) { return S::SimtR8C8; }
        return S::MmaR64C128;
    }
    if (problem.rows == 4608 && problem.k == 4608) {
        if (problem.cols <= 8) { return S::SimtR8C4; }
        if (problem.cols <= 11) { return S::MmaR32C128; }
        if (problem.cols == 12) { return S::SimtR8C4; }
        if (problem.cols <= 256) { return S::MmaR32C128; }
        return S::MmaR64C128;
    }
    if (problem.rows == 5120 && problem.k == 4608) {
        if (problem.cols <= 4) { return S::SimtR8C4; }
        if (problem.cols == 5) { return S::SimtR8C8; }
        return S::MmaR64C128;
    }
    if (problem.rows == 2048 && problem.k == 4608) {
        if (problem.cols <= 14 || problem.cols == 16 || problem.cols == 20 || problem.cols == 24 ||
            problem.cols == 28 || problem.cols == 32) {
            return S::SimtR8C4;
        }
        if (problem.cols <= 871) { return S::MmaR32C128; }
        return S::MmaR64C128;
    }
    if (problem.rows == 1024) {
        if (problem.cols <= 4) { return S::SimtR8C4; }
        if (problem.cols <= 16) { return S::SimtR8C8; }
        return S::MmaR32C128;
    }
    if (problem.cols <= 4) { return S::SimtR8C4; }
    if (problem.cols <= 16) { return S::SimtR8C8; }
    return S::MmaR64C128;
}

V expected_variant(S schedule, const W8Problem& problem) {
    const int rows  = schedule == S::MmaR32C128 ? 32 : (schedule == S::MmaR64C128 ? 64 : 8);
    const int cols  = schedule == S::SimtR8C4 ? 4 : (schedule == S::SimtR8C8 ? 8 : 128);
    const bool full = problem.rows % rows == 0 && problem.cols % cols == 0 &&
                      (!ninfer::ops::detail::w8_schedule_uses_mma(schedule) ||
                       (problem.k == problem.padded_k && problem.k % 64 == 0));
    return full ? V::Full : V::Predicated;
}

void expect_plan(const std::string& label, const W8Problem& problem) {
    if (!ninfer::ops::detail::w8_rowsplit_admits(problem)) {
        fail(label, "production admission rejected the problem");
        return;
    }
    const S schedule = expected_schedule(problem);
    const W8Plan expected{schedule, expected_variant(schedule, problem)};
    try {
        const W8Plan actual = ninfer::ops::detail::w8_rowsplit_resolve_plan(problem);
        if (actual.schedule != expected.schedule || actual.variant != expected.variant) {
            fail(label, "expected " + plan_name(expected) + ", got " + plan_name(actual));
        }
        if (!ninfer::ops::detail::w8_candidate_is_legal(actual.schedule, actual.variant, problem)) {
            fail(label, "resolver returned a physically illegal plan");
        }
    } catch (const std::exception& error) {
        fail(label, std::string("resolver threw: ") + error.what());
    }
}

void route_boundaries() {
    constexpr std::array<W8Problem, 60> cases{{
        {5120, 10240, 10240, 1},       {5120, 10240, 10240, 4},  {5120, 10240, 10240, 5},
        {5120, 10240, 10240, 16},      {5120, 10240, 10240, 17}, {5120, 10240, 10240, 128},
        {5120, 10240, 10240, 8388480}, {14336, 5120, 5120, 4},   {14336, 5120, 5120, 5},
        {14336, 5120, 5120, 8},        {14336, 5120, 5120, 9},   {1024, 5120, 5120, 4},
        {1024, 5120, 5120, 5},         {1024, 5120, 5120, 16},   {1024, 5120, 5120, 17},
        {1024, 5120, 5120, 128},       {6144, 5120, 5120, 16},   {6144, 5120, 5120, 17},
        {5120, 6144, 6144, 16},        {5120, 6144, 6144, 17},   {34816, 5120, 5120, 4},
        {34816, 5120, 5120, 8},        {34816, 5120, 5120, 9},   {5120, 17408, 17408, 16},
        {5120, 17408, 17408, 17},      {4608, 4608, 4608, 1},    {4608, 4608, 4608, 8},
        {4608, 4608, 4608, 9},         {4608, 4608, 4608, 11},   {4608, 4608, 4608, 12},
        {4608, 4608, 4608, 13},        {4608, 4608, 4608, 256},  {4608, 4608, 4608, 257},
        {4608, 4608, 4608, 32768},     {5120, 4608, 4608, 4},    {5120, 4608, 4608, 5},
        {5120, 4608, 4608, 6},         {5120, 4608, 4608, 127},  {5120, 4608, 4608, 128},
        {5120, 4608, 4608, 32768},     {2048, 4608, 4608, 1},    {2048, 4608, 4608, 14},
        {2048, 4608, 4608, 15},        {2048, 4608, 4608, 16},   {2048, 4608, 4608, 17},
        {2048, 4608, 4608, 19},        {2048, 4608, 4608, 20},   {2048, 4608, 4608, 21},
        {2048, 4608, 4608, 23},        {2048, 4608, 4608, 24},   {2048, 4608, 4608, 25},
        {2048, 4608, 4608, 27},        {2048, 4608, 4608, 28},   {2048, 4608, 4608, 29},
        {2048, 4608, 4608, 31},        {2048, 4608, 4608, 32},   {2048, 4608, 4608, 33},
        {2048, 4608, 4608, 871},       {2048, 4608, 4608, 872},  {2048, 4608, 4608, 32768},
    }};
    for (const W8Problem& problem : cases) { expect_plan("route boundary", problem); }
}

void rejection_contract() {
    constexpr std::array<W8Problem, 9> rejected{{
        {5120, 10240, 10240, 0},
        {5120, 10240, 10240, 8388481},
        {4608, 4608, 4608, 32769},
        {2048, 4608, 4608, 32769},
        {1024, 5120, 5248, 17},
        {7168, 5120, 5120, 1},
        {2048, 4096, 4096, 1},
        {5120, 10240, 10368, 17},
        {5120, 2048, 2048, 1},
    }};
    for (const W8Problem& problem : rejected) {
        if (ninfer::ops::detail::w8_rowsplit_admits(problem)) {
            fail("rejection", "production admission accepted an unregistered problem");
        }
        try {
            const W8Plan plan = ninfer::ops::detail::w8_rowsplit_resolve_plan(problem);
            fail("rejection", "resolver returned " + plan_name(plan));
        } catch (const std::invalid_argument&) {
        } catch (const std::exception& error) {
            fail("rejection", std::string("resolver threw wrong exception: ") + error.what());
        }
    }
}

void pair_contract() {
    using PS = W8PairScheduleId;
    constexpr std::array<std::int32_t, 9> cols{{
        1,
        4,
        5,
        56,
        57,
        127,
        128,
        2048,
        8388480,
    }};
    for (const std::int32_t value : cols) {
        const W8PairProblem problem{1024, 5120, 5120, value};
        if (!ninfer::ops::detail::w8_pair_admits(problem)) {
            fail("pair admission", "rejected admitted column count");
            continue;
        }
        const W8PairPlan plan = ninfer::ops::detail::w8_pair_resolve_plan(problem);
        const PS expected     = value <= 4    ? PS::TwoSimtR8C4
                                : value <= 56 ? PS::TwoSimtR8C8
                                              : PS::DualMmaR32C128;
        if (plan.schedule != expected) {
            fail("pair route", "unexpected schedule at C=" + std::to_string(value));
        }
        const W8Problem base{problem.rows, problem.k, problem.padded_k, problem.cols};
        const S base_schedule = expected == PS::TwoSimtR8C4   ? S::SimtR8C4
                                : expected == PS::TwoSimtR8C8 ? S::SimtR8C8
                                                              : S::MmaR32C128;
        if (plan.variant != expected_variant(base_schedule, base)) {
            fail("pair variant", "unexpected variant at C=" + std::to_string(value));
        }
        if (plan.workspace_bytes != 0 ||
            plan.performance_qualified != (value <= ninfer::ops::detail::kW8PairQualifiedCols)) {
            fail("pair metadata",
                 "unexpected workspace/qualification at C=" + std::to_string(value));
        }
    }

    for (const W8PairProblem problem :
         {W8PairProblem{1024, 5120, 5120, 0}, W8PairProblem{1024, 5120, 5120, 8388481},
          W8PairProblem{6144, 5120, 5120, 57}}) {
        if (ninfer::ops::detail::w8_pair_admits(problem)) {
            fail("pair rejection", "accepted an unregistered pair problem");
        }
        try {
            (void)ninfer::ops::detail::w8_pair_resolve_plan(problem);
            fail("pair rejection", "resolved an unregistered pair problem");
        } catch (const std::invalid_argument&) {}
    }
}

} // namespace

int main() {
    route_boundaries();
    rejection_contract();
    pair_contract();

    std::cout << (failures == 0 ? "OK" : "FAIL") << " W8 Linear production plan\n";
    return failures == 0 ? 0 : 1;
}
