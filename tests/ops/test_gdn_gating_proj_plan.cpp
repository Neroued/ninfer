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
    constexpr std::array<std::int32_t, 16> boundaries{
        1,
        2,
        8,
        9,
        127,
        128,
        1024,
        1025,
        2048,
        2049,
        4095,
        4096,
        4097,
        8192,
        ninfer::ops::detail::kBf16GdnGatingMaxCols - 1,
        ninfer::ops::detail::kBf16GdnGatingMaxCols,
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
        if (plan.performance_qualified !=
            (cols <= ninfer::ops::detail::kBf16GdnGatingQualifiedCols)) {
            std::cerr << "wrong BF16 GDN gating qualification C=" << cols << '\n';
            ++failures;
        }
    }

    expect_invalid("C0",
                   [] { (void)ninfer::ops::detail::bf16_gdn_gating_resolve_plan({48, 5120, 0}); });
    expect_invalid("unsupported heads",
                   [] { (void)ninfer::ops::detail::bf16_gdn_gating_resolve_plan({47, 5120, 1}); });
}

void workspace_tests() {
    struct Case {
        std::int32_t capacity;
        std::size_t bytes;
    };

    constexpr std::array<Case, 12> cases{{
        {1, 0},
        {2, 7'680},
        {8, 30'720},
        {9, 30'720},
        {10, 30'720},
        {128, 393'216},
        {1024, 3'145'728},
        {1025, 3'145'728},
        {2048, 3'145'728},
        {4096, 3'145'728},
        {4097, 3'145'728},
        {ninfer::ops::detail::kBf16GdnGatingMaxCols, 3'145'728},
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
    workspace_tests();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " BF16 GDN gating projection plan\n";
    return failures == 0 ? 0 : 1;
}
