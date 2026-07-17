#include "ninfer/ops/gdn_input_proj.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

using ninfer::ops::detail::Q4KernelVariant;
using ninfer::ops::detail::Q4Q5AttnInputProblem;
using ninfer::ops::detail::Q4Q5AttnInputScheduleId;
using ninfer::ops::detail::Q4Q5GdnInputProblem;
using ninfer::ops::detail::Q4Q5GdnInputScheduleId;
using ninfer::ops::detail::Q4ScheduleId;
using ninfer::ops::detail::Q5ScheduleId;

int failures = 0;

template <class Fn>
void expect_invalid(const char* label, Fn&& fn) {
    try {
        fn();
        std::cerr << label << ": expected invalid_argument\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
}

void attn_route_tests() {
    constexpr std::array<std::int32_t, 10> boundaries{
        1, 2, 16, 17, 127, 128, 129, 1024, 1025, 2048,
    };
    for (const std::int32_t cols : boundaries) {
        const Q4Q5AttnInputProblem problem{5120, 6144, 1024, 5120, cols};
        const auto plan         = ninfer::ops::detail::q4_q5_attn_input_resolve_plan(problem);
        const bool parent_split = cols <= 16;
        const auto expected     = parent_split
                                      ? Q4Q5AttnInputScheduleId::ParentSplitFixed
                                      : Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR64C128;
        if (plan.schedule != expected || plan.parent_split.has_value() != parent_split ||
            plan.workspace_bytes != 0) {
            std::cerr << "wrong attention input route C=" << cols << '\n';
            ++failures;
        }
        const Q4KernelVariant expected_variant =
            parent_split
                ? Q4KernelVariant::None
                : ((cols % 128) == 0 ? Q4KernelVariant::Full : Q4KernelVariant::Predicated);
        if (plan.grouped_variant != expected_variant) {
            std::cerr << "wrong attention input variant C=" << cols << '\n';
            ++failures;
        }
    }
    for (std::int32_t cols = 1; cols <= 16; ++cols) {
        const auto plan =
            ninfer::ops::detail::q4_q5_attn_input_resolve_plan({5120, 6144, 1024, 5120, cols});
        const Q4ScheduleId expected_q4 =
            cols == 1 ? Q4ScheduleId::GemvR1W8Direct
                      : ((cols <= 7 || (cols >= 9 && cols <= 15)) ? Q4ScheduleId::SimtR8C4
                                                                  : Q4ScheduleId::SimtR8C8);
        const Q5ScheduleId expected_q5 =
            cols == 1 ? Q5ScheduleId::GemvR16S2X
                      : (cols <= 6 ? Q5ScheduleId::SimtSplit4Exact : Q5ScheduleId::SimtR8C4);
        if (!plan.parent_split.has_value() ||
            plan.parent_split->query_key.schedule != expected_q4 ||
            plan.parent_split->gate_value.schedule != expected_q5) {
            std::cerr << "wrong Attention parent subplans C=" << cols << '\n';
            ++failures;
        }
    }
    expect_invalid("attention C0", [] {
        (void)ninfer::ops::detail::q4_q5_attn_input_resolve_plan({5120, 6144, 1024, 5120, 0});
    });
    expect_invalid("attention unsupported shape", [] {
        (void)ninfer::ops::detail::q4_q5_attn_input_resolve_plan({5120, 6144, 2048, 5120, 1});
    });
}

void gdn_route_tests() {
    constexpr std::array<std::int32_t, 10> boundaries{
        1, 2, 16, 17, 127, 128, 129, 1024, 1025, 2048,
    };
    for (const std::int32_t cols : boundaries) {
        const Q4Q5GdnInputProblem problem{5120, 4096, 6144, 10240, 5120, cols};
        const auto plan        = ninfer::ops::detail::q4_q5_gdn_input_resolve_plan(problem);
        const bool independent = cols <= 16;
        const auto expected    = independent ? Q4Q5GdnInputScheduleId::IndependentDirectFixed
                                             : Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128;
        if (plan.schedule != expected || plan.independent.has_value() != independent ||
            plan.workspace_bytes != 0) {
            std::cerr << "wrong GDN input route C=" << cols << '\n';
            ++failures;
        }
        const Q4KernelVariant expected_variant =
            independent ? Q4KernelVariant::None
                        : ((cols % 128) == 0 ? Q4KernelVariant::Full : Q4KernelVariant::Predicated);
        if (plan.grouped_variant != expected_variant) {
            std::cerr << "wrong GDN input variant C=" << cols << '\n';
            ++failures;
        }
    }
    expect_invalid("GDN C0", [] {
        (void)ninfer::ops::detail::q4_q5_gdn_input_resolve_plan({5120, 4096, 6144, 10240, 5120, 0});
    });
    expect_invalid("GDN unsupported shape", [] {
        (void)ninfer::ops::detail::q4_q5_gdn_input_resolve_plan({5120, 4096, 6144, 10241, 5120, 1});
    });
}

void workspace_tests() {
    struct Case {
        std::int32_t capacity;
        std::size_t bytes;
    };

    constexpr std::array<Case, 8> cases{{
        {1, 0},
        {2, 0},
        {16, 0},
        {17, 0},
        {128, 0},
        {1024, 0},
        {1025, 0},
        {2048, 0},
    }};
    for (const Case test : cases) {
        const std::size_t actual =
            ninfer::ops::gdn_input_proj_workspace_bytes(4096, 6144, test.capacity);
        if (actual != test.bytes) {
            std::cerr << "GDN input workspace C=" << test.capacity << ": expected " << test.bytes
                      << ", got " << actual << '\n';
            ++failures;
        }
    }
    expect_invalid("workspace unsupported rows",
                   [] { (void)ninfer::ops::gdn_input_proj_workspace_bytes(4095, 6144, 1); });
    expect_invalid("workspace C0",
                   [] { (void)ninfer::ops::gdn_input_proj_workspace_bytes(4096, 6144, 0); });
}

} // namespace

int main() {
    attn_route_tests();
    gdn_route_tests();
    workspace_tests();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " Q4/Q5 input projection plans\n";
    return failures == 0 ? 0 : 1;
}
