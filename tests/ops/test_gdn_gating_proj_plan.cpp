#include "ninfer/ops/gdn_gating_proj.h"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

using ninfer::ops::detail::Bf16GdnGatingScheduleId;
using ninfer::ops::detail::Bf16GdnGatingTokenVariant;

int failures = 0;

template <class Fn>
void expect_invalid(const char* label, Fn&& fn) {
    try {
        fn();
        std::cerr << label << ": expected invalid_argument\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
}

Bf16GdnGatingScheduleId expected_schedule(std::int32_t cols) {
    if (cols == 1) { return Bf16GdnGatingScheduleId::GemvPairedRows; }
    if (cols <= 8) { return Bf16GdnGatingScheduleId::SmallTSplit10; }
    if (cols <= 1024) { return Bf16GdnGatingScheduleId::MmaCooperativeSplit8; }
    if (cols <= 2048) { return Bf16GdnGatingScheduleId::MmaCooperativeSplit4; }
    if (cols <= 4096) { return Bf16GdnGatingScheduleId::MmaCooperativeSplit2; }
    return Bf16GdnGatingScheduleId::MmaUnsplit;
}

std::int32_t expected_split(std::int32_t cols) {
    if (cols == 1 || cols > 4096) { return 1; }
    if (cols <= 8) { return 10; }
    if (cols <= 1024) { return 8; }
    if (cols <= 2048) { return 4; }
    return 2;
}

void route_tests() {
    using S = Bf16GdnGatingScheduleId;
    constexpr std::array<std::int32_t, 14> boundaries{
        1, 2, 8, 9, 127, 128, 1024, 1025, 2048, 2049, 4095, 4096, 4097, 8192,
    };
    for (const std::int32_t cols : boundaries) {
        const auto plan = ninfer::ops::detail::bf16_gdn_gating_resolve_plan({48, 5120, cols});
        if (plan.schedule != expected_schedule(cols)) {
            std::cerr << "wrong BF16 GDN gating route C=" << cols << '\n';
            ++failures;
        }
        const bool mma = cols >= 9;
        const Bf16GdnGatingTokenVariant expected_variant =
            !mma ? Bf16GdnGatingTokenVariant::None
                 : ((cols % 128) == 0 ? Bf16GdnGatingTokenVariant::Full
                                      : Bf16GdnGatingTokenVariant::Predicated);
        if (plan.token_variant != expected_variant) {
            std::cerr << "wrong BF16 GDN gating variant C=" << cols << '\n';
            ++failures;
        }
        const std::int32_t split = expected_split(cols);
        const std::size_t expected_workspace =
            split > 1 ? static_cast<std::size_t>(split) * cols * 96u * sizeof(float) : 0;
        if (plan.workspace_bytes != expected_workspace) {
            std::cerr << "wrong BF16 GDN gating workspace C=" << cols << '\n';
            ++failures;
        }
    }

    expect_invalid("C0",
                   [] { (void)ninfer::ops::detail::bf16_gdn_gating_resolve_plan({48, 5120, 0}); });
    expect_invalid("unsupported heads",
                   [] { (void)ninfer::ops::detail::bf16_gdn_gating_resolve_plan({47, 5120, 1}); });

    struct CooperativeBoundary {
        S schedule;
        std::int32_t last_legal_cols;
        const char* label;
    };

    constexpr std::array<CooperativeBoundary, 3> cooperative_boundaries{{
        {S::MmaCooperativeSplit8, 1792, "27B split8"},
        {S::MmaCooperativeSplit4, 3584, "27B split4"},
        {S::MmaCooperativeSplit2, 7168, "27B split2"},
    }};
    for (const CooperativeBoundary& boundary : cooperative_boundaries) {
        (void)ninfer::ops::detail::bf16_gdn_gating_resolve_candidate(
            boundary.schedule, {48, 5120, boundary.last_legal_cols});
        expect_invalid(boundary.label, [&boundary] {
            (void)ninfer::ops::detail::bf16_gdn_gating_resolve_candidate(
                boundary.schedule, {48, 5120, boundary.last_legal_cols + 1});
        });
    }
}

void route35_tests() {
    using S = Bf16GdnGatingScheduleId;
    constexpr std::array<std::int32_t, 10> boundaries{1,    127,  128,  1024, 1025,
                                                      2048, 2049, 4096, 4097, 8192};
    for (const std::int32_t cols : boundaries) {
        const ninfer::ops::detail::Bf16GdnGatingProblem problem{32, 2048, cols};
        const bool admitted = cols >= 1;
        if (ninfer::ops::detail::bf16_gdn_gating_admits(problem) != admitted) {
            std::cerr << "wrong 35B GDN gating admission C=" << cols << '\n';
            ++failures;
            continue;
        }
        if (!admitted) { continue; }
        const auto plan          = ninfer::ops::detail::bf16_gdn_gating_resolve_plan(problem);
        const S expected         = cols <= 127    ? S::MmaCooperativeSplit16
                                   : cols <= 1024 ? S::MmaCooperativeSplit8
                                   : cols <= 2048 ? S::MmaCooperativeSplit4
                                   : cols <= 4096 ? S::MmaCooperativeSplit2
                                                  : S::MmaUnsplit;
        const std::int32_t split = cols <= 127    ? 16
                                   : cols <= 1024 ? 8
                                   : cols <= 2048 ? 4
                                   : cols <= 4096 ? 2
                                                  : 1;
        const std::size_t workspace =
            split > 1 ? static_cast<std::size_t>(split) * cols * 64u * sizeof(float) : 0;
        const auto variant = (cols % 64) == 0 ? Bf16GdnGatingTokenVariant::Full
                                              : Bf16GdnGatingTokenVariant::Predicated;
        if (plan.schedule != expected || plan.token_variant != variant ||
            plan.workspace_bytes != workspace) {
            std::cerr << "wrong 35B GDN gating route C=" << cols << '\n';
            ++failures;
        }
    }

    struct CooperativeBoundary {
        S schedule;
        std::int32_t split;
        std::int32_t last_legal_cols;
        const char* label;
    };

    constexpr std::array<CooperativeBoundary, 5> cooperative_boundaries{{
        {S::MmaCooperativeSplit32, 32, 320, "split32"},
        {S::MmaCooperativeSplit16, 16, 1344, "split16"},
        {S::MmaCooperativeSplit8, 8, 2688, "split8"},
        {S::MmaCooperativeSplit4, 4, 5440, "split4"},
        {S::MmaCooperativeSplit2, 2, 10880, "split2"},
    }};
    for (const CooperativeBoundary& boundary : cooperative_boundaries) {
        const auto plan = ninfer::ops::detail::bf16_gdn_gating_resolve_candidate(
            boundary.schedule, {32, 2048, boundary.last_legal_cols});
        const std::size_t expected_workspace = static_cast<std::size_t>(boundary.split) *
                                               boundary.last_legal_cols * 64u * sizeof(float);
        if (plan.workspace_bytes != expected_workspace) {
            std::cerr << "wrong 35B " << boundary.label << " candidate workspace\n";
            ++failures;
        }
        expect_invalid(boundary.label, [&boundary] {
            (void)ninfer::ops::detail::bf16_gdn_gating_resolve_candidate(
                boundary.schedule, {32, 2048, boundary.last_legal_cols + 1});
        });
    }
    (void)ninfer::ops::detail::bf16_gdn_gating_resolve_candidate(S::SimtWarpRowC4,
                                                                 {32, 2048, 262140});
    expect_invalid("35B simt-c4 grid.y too large", [] {
        (void)ninfer::ops::detail::bf16_gdn_gating_resolve_candidate(
            Bf16GdnGatingScheduleId::SimtWarpRowC4, {32, 2048, 262141});
    });
    (void)ninfer::ops::detail::bf16_gdn_gating_resolve_candidate(S::SimtWarpRowC8,
                                                                 {32, 2048, 524280});
    expect_invalid("35B simt-c8 grid.y too large", [] {
        (void)ninfer::ops::detail::bf16_gdn_gating_resolve_candidate(
            Bf16GdnGatingScheduleId::SimtWarpRowC8, {32, 2048, 524281});
    });
    expect_invalid("35B legacy small-T candidate", [] {
        (void)ninfer::ops::detail::bf16_gdn_gating_resolve_candidate(
            Bf16GdnGatingScheduleId::SmallTSplit10, {32, 2048, 2});
    });
}

void workspace_tests() {
    struct Case {
        std::int32_t capacity;
        std::size_t bytes;
    };

    constexpr std::array<Case, 12> cases{{
        {1, 4'096},
        {2, 8'192},
        {8, 32'768},
        {9, 36'864},
        {10, 40'960},
        {128, 520'192},
        {1024, 3'145'728},
        {1025, 3'145'728},
        {2048, 3'145'728},
        {4096, 3'145'728},
        {4097, 3'145'728},
        {8192, 3'145'728},
    }};
    for (const Case test : cases) {
        const std::size_t actual = ninfer::ops::gdn_gating_proj_workspace_bytes(test.capacity);
        if (actual != test.bytes) {
            std::cerr << "workspace C=" << test.capacity << ": expected " << test.bytes << ", got "
                      << actual << '\n';
            ++failures;
        }
    }
    expect_invalid("workspace C0", [] { (void)ninfer::ops::gdn_gating_proj_workspace_bytes(0); });
}

} // namespace

int main() {
    route_tests();
    route35_tests();
    workspace_tests();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " BF16 GDN gating projection plan\n";
    return failures == 0 ? 0 : 1;
}
