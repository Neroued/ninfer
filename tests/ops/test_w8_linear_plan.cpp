#include "ops/common/token_slices.h"
#include "ops/linear/w8/w8_rowsplit_plan.h"
#include "ops/linear_pair/w8/w8_pair_plan.h"

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using ninfer::ops::detail::W8KernelVariant;
using ninfer::ops::detail::W8PairPlan;
using ninfer::ops::detail::W8PairProblem;
using ninfer::ops::detail::W8PairScheduleId;
using ninfer::ops::detail::W8Plan;
using ninfer::ops::detail::W8Problem;
using ninfer::ops::detail::W8ScheduleId;
using ninfer::ops::detail::W8TailPolicy;

namespace {

using S = W8ScheduleId;
using V = W8KernelVariant;
using P = W8TailPolicy;

int failures = 0;

void fail(const std::string& label, const std::string& message) {
    std::cerr << "FAIL " << label << ": " << message << '\n';
    ++failures;
}

std::string plan_name(W8Plan plan) {
    return std::string(ninfer::ops::detail::w8_schedule_name(plan.schedule)) + "." +
           ninfer::ops::detail::w8_kernel_variant_name(plan.variant) +
           (plan.tail_policy == P::ConditioningExact ? ".conditioning_exact_tail" : "");
}

S expected_schedule(const W8Problem& problem) {
    if (problem.rows == 2048 && problem.k == 16384) {
        if (problem.cols == 1) { return S::DecodeR4; }
        if (problem.cols <= 32) { return S::SplitKMmaExactT; }
        if (problem.cols <= 88) { return S::SplitKMma32PlusTail; }
        if (problem.cols <= 96) { return S::SplitKMediumC96; }
        if (problem.cols <= 128) { return S::SplitKMediumC128; }
        if (problem.cols <= 144) { return S::SplitKMediumC144; }
        if (problem.cols <= 255) { return S::MmaR32C128; }
        if (problem.cols <= 384) { return S::MmaR32C64; }
        if (problem.cols <= 480) { return S::MmaR32C96; }
        if (problem.cols == 481) { return S::MmaR32C96; }
        if (problem.cols <= 668) { return S::MmaR32C128; }
        if (problem.cols <= 673) { return S::MmaR48C96; }
        if (problem.cols <= 704) { return S::MmaR48C64; }
        if (problem.cols <= 784) { return S::MmaR48C112; }
        if (problem.cols <= 912) { return S::MmaR48C128; }
        if (problem.cols <= 1007) { return S::MmaR64C96; }
        if (problem.cols == 1008) { return S::MmaR64C112; }
        if (problem.cols <= 1119) { return S::MmaR64C128; }
        if (problem.cols == 1120) { return S::MmaR64C112; }
        if (problem.cols <= 1280) { return S::MmaR64C128; }
        if (problem.cols <= 1313) { return S::MmaR64C128; }
        if (problem.cols <= 1344) { return S::MmaR128C64; }
        if (problem.cols <= 1500) { return S::MmaR96C96; }
        if (problem.cols <= 1745) { return S::MmaR128C80; }
        if (problem.cols <= 1791) { return S::MmaR48C128; }
        if (problem.cols == 1792) { return S::MmaR64C128; }
        if (problem.cols <= 1919) { return S::MmaR48C128; }
        if (problem.cols == 1920) { return S::MmaR64C128; }
        if (problem.cols <= 1953) { return S::MmaR64C128; }
        if (problem.cols <= 2048) { return S::MmaR64C96; }
        if (problem.cols <= 2112) { return S::MmaR96C96; }
        return S::MmaR64C128;
    }
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
    if (problem.rows == 2048 && problem.k == 4096) {
        if (problem.cols <= 56) { return S::SimtR8C4; }
        if (problem.cols <= 895) { return S::MmaR32C128; }
        return S::MmaR64C128;
    }
    if (problem.rows == 12288 && problem.k == 2048) {
        return problem.cols <= 16 ? S::SimtR8C4 : S::MmaR64C128;
    }
    if (problem.rows == 9216 && problem.k == 2048) {
        if (problem.cols <= 13) { return S::SimtR8C4; }
        if (problem.cols <= 128) { return S::MmaR32C128; }
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
    if (ninfer::ops::detail::w8_candidate_is_legal(schedule, V::None, problem)) { return V::None; }
    if (ninfer::ops::detail::w8_candidate_is_legal(schedule, V::Full, problem)) { return V::Full; }
    return V::Predicated;
}

P expected_tail_policy(const W8Problem& problem) {
    if (problem.rows != 2048 || problem.k != 16384) { return P::Homogeneous; }
    switch (problem.cols) {
    case 481:
    case 673:
        return P::ConditioningExact;
    default:
        return (problem.cols >= 641 && problem.cols <= 668) ||
                       (problem.cols >= 897 && problem.cols <= 912) ||
                       (problem.cols >= 961 && problem.cols <= 1007) ||
                       (problem.cols >= 1281 && problem.cols <= 1313) ||
                       (problem.cols >= 1441 && problem.cols <= 1500) ||
                       (problem.cols >= 1681 && problem.cols <= 1745) ||
                       (problem.cols >= 1921 && problem.cols <= 1953) ||
                       (problem.cols >= 2017 && problem.cols <= 2048)
                   ? P::ConditioningExact
                   : P::Homogeneous;
    }
}

void expect_plan(const std::string& label, const W8Problem& problem) {
    if (!ninfer::ops::detail::w8_rowsplit_admits(problem)) {
        fail(label, "production admission rejected the problem");
        return;
    }
    const S schedule = expected_schedule(problem);
    const W8Plan expected{schedule, expected_variant(schedule, problem),
                          expected_tail_policy(problem)};
    try {
        const W8Plan actual = ninfer::ops::detail::w8_rowsplit_resolve_plan(problem);
        if (actual.schedule != expected.schedule || actual.variant != expected.variant ||
            actual.tail_policy != expected.tail_policy) {
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
    constexpr std::array<W8Problem, 78> cases{{
        {5120, 10240, 10240, 1},  {5120, 10240, 10240, 4},   {5120, 10240, 10240, 5},
        {5120, 10240, 10240, 16}, {5120, 10240, 10240, 17},  {5120, 10240, 10240, 128},
        {14336, 5120, 5120, 4},   {14336, 5120, 5120, 5},    {14336, 5120, 5120, 8},
        {14336, 5120, 5120, 9},   {1024, 5120, 5120, 4},     {1024, 5120, 5120, 5},
        {1024, 5120, 5120, 16},   {1024, 5120, 5120, 17},    {1024, 5120, 5120, 128},
        {6144, 5120, 5120, 16},   {6144, 5120, 5120, 17},    {5120, 6144, 6144, 16},
        {5120, 6144, 6144, 17},   {34816, 5120, 5120, 4},    {34816, 5120, 5120, 8},
        {34816, 5120, 5120, 9},   {5120, 17408, 17408, 16},  {5120, 17408, 17408, 17},
        {4608, 4608, 4608, 1},    {4608, 4608, 4608, 8},     {4608, 4608, 4608, 9},
        {4608, 4608, 4608, 11},   {4608, 4608, 4608, 12},    {4608, 4608, 4608, 13},
        {4608, 4608, 4608, 256},  {4608, 4608, 4608, 257},   {4608, 4608, 4608, 32768},
        {5120, 4608, 4608, 4},    {5120, 4608, 4608, 5},     {5120, 4608, 4608, 6},
        {5120, 4608, 4608, 127},  {5120, 4608, 4608, 128},   {5120, 4608, 4608, 32768},
        {2048, 4608, 4608, 1},    {2048, 4608, 4608, 14},    {2048, 4608, 4608, 15},
        {2048, 4608, 4608, 16},   {2048, 4608, 4608, 17},    {2048, 4608, 4608, 19},
        {2048, 4608, 4608, 20},   {2048, 4608, 4608, 21},    {2048, 4608, 4608, 23},
        {2048, 4608, 4608, 24},   {2048, 4608, 4608, 25},    {2048, 4608, 4608, 27},
        {2048, 4608, 4608, 28},   {2048, 4608, 4608, 29},    {2048, 4608, 4608, 31},
        {2048, 4608, 4608, 32},   {2048, 4608, 4608, 33},    {2048, 4608, 4608, 871},
        {2048, 4608, 4608, 872},  {2048, 4608, 4608, 32768}, {2048, 4096, 4096, 1},
        {2048, 4096, 4096, 56},   {2048, 4096, 4096, 57},    {2048, 4096, 4096, 895},
        {2048, 4096, 4096, 896},  {2048, 4096, 4096, 1024},  {12288, 2048, 2048, 1},
        {12288, 2048, 2048, 16},  {12288, 2048, 2048, 17},   {12288, 2048, 2048, 1024},
        {9216, 2048, 2048, 1},    {9216, 2048, 2048, 13},    {9216, 2048, 2048, 14},
        {9216, 2048, 2048, 128},  {9216, 2048, 2048, 129},   {9216, 2048, 2048, 1024},
        {2048, 4096, 4096, 1025}, {12288, 2048, 2048, 1025}, {9216, 2048, 2048, 1025},
    }};
    for (const W8Problem& problem : cases) { expect_plan("route boundary", problem); }

    constexpr std::int32_t conditioning_cols[]{
        1,   2,    31,   32,   33,   88,   89,   96,   97,   128,  129,  144,
        145, 255,  256,  384,  385,  480,  481,  482,  640,  641,  668,  669,
        672, 673,  674,  704,  705,  784,  785,  896,  897,  912,  913,  960,
        961, 1007, 1008, 1009, 1119, 1120, 1121, 1280, 1281, 1313, 1314, 1344,
        1345, 1440, 1441, 1500, 1501, 1680, 1681, 1745, 1746, 1791, 1792, 1793,
        1919, 1920, 1921, 1953, 1954, 2016, 2017, 2048, 2049, 2112, 2113, 2176,
        4096, 8192,
    };
    for (const std::int32_t cols : conditioning_cols) {
        expect_plan("conditioning route boundary", {2048, 16384, 16384, cols});
    }
}

void rejection_contract() {
    constexpr std::array<W8Problem, 8> rejected{{
        {5120, 10240, 10240, 0},
        {4608, 4608, 4608, 32769},
        {5120, 4608, 4608, 32769},
        {2048, 4608, 4608, 32769},
        {1024, 5120, 5248, 17},
        {7168, 5120, 5120, 1},
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
        4096,
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
        if (plan.workspace_bytes != 0) {
            fail("pair metadata", "unexpected workspace at C=" + std::to_string(value));
        }
    }

    for (const W8PairProblem problem :
         {W8PairProblem{1024, 5120, 5120, 0}, W8PairProblem{6144, 5120, 5120, 57}}) {
        if (ninfer::ops::detail::w8_pair_admits(problem)) {
            fail("pair rejection", "accepted an unregistered pair problem");
        }
        try {
            (void)ninfer::ops::detail::w8_pair_resolve_plan(problem);
            fail("pair rejection", "resolved an unregistered pair problem");
        } catch (const std::invalid_argument&) {}
    }
}

void token_slice_boundaries() {
    constexpr std::int32_t kColumnsPerBlock = 128;
    constexpr std::int32_t kSliceCapacity = kColumnsPerBlock * ninfer::ops::detail::kCudaGridYLimit;

    const auto collect = [](std::int32_t tokens) {
        std::vector<std::pair<std::int32_t, std::int32_t>> slices;
        ninfer::ops::detail::for_each_token_slice(
            tokens, kColumnsPerBlock, [&slices](std::int32_t offset, std::int32_t count) {
                slices.emplace_back(offset, count);
            });
        return slices;
    };

    if (collect(1024) != std::vector<std::pair<std::int32_t, std::int32_t>>{{0, 1024}}) {
        fail("token slicing default target", "1024 should require one launch");
    }
    if (collect(kSliceCapacity) !=
        std::vector<std::pair<std::int32_t, std::int32_t>>{{0, kSliceCapacity}}) {
        fail("token slicing last single launch", "grid-y capacity should require one launch");
    }
    if (collect(kSliceCapacity + 1) != std::vector<std::pair<std::int32_t, std::int32_t>>{
                                           {0, kSliceCapacity}, {kSliceCapacity, 1}}) {
        fail("token slicing first split launch", "capacity+1 should produce two exact slices");
    }

    std::int64_t expected_offset = 0;
    ninfer::ops::detail::for_each_token_slice(
        std::numeric_limits<std::int32_t>::max(), kColumnsPerBlock,
        [&expected_offset](std::int32_t offset, std::int32_t count) {
            if (offset != expected_offset || count <= 0 || count > kSliceCapacity) {
                fail("token slicing int32 domain", "slices are not contiguous or launchable");
            }
            expected_offset += count;
        });
    if (expected_offset != std::numeric_limits<std::int32_t>::max()) {
        fail("token slicing int32 domain", "slices do not cover the complete logical T");
    }
}

} // namespace

int main() {
    route_boundaries();
    rejection_contract();
    pair_contract();
    token_slice_boundaries();

    std::cout << (failures == 0 ? "OK" : "FAIL") << " W8 Linear production plan\n";
    return failures == 0 ? 0 : 1;
}
