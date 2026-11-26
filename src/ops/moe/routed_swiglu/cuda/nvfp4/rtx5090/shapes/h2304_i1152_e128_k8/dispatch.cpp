#include "ops/moe/routed_swiglu/cuda/dispatch.h"

#include <array>

namespace ninfer::ops::detail::moe_routed {
namespace {
using softcap::Schedule;

struct TokenChoice {
    int first;
    Schedule a16, a4;
};

// RTX 5090 cold Graph body over five routing distributions. Grouped comparisons
// use their geometric mean unless a distribution is named.
// At T=1, complete Down measured 29.7 us vs 30.0 us for split on Uniform and saved a launch.
// The split A16 band starts at T=2 and wins Uniform T=8 at 152.6 vs 154.3 us.
//      Reuse dispatch at T=2 gave 58.4 us on Uniform and 40.0 us on Hot-8,
//      vs 62.8 and 35.8 us for short FP4. At T=8 it held both latency targets.
//     Mean-load cutoffs 1.25, 1.5, and 2 measured 44.6, 44.0, and 45.5 us on
//     Hot-8 T=8. Grouped FP4 uses the 1.5 cutoff.
// At T=9, short FP4 averaged 125.5 us over T=9..16 vs 134.7 us for reuse dispatch.
// At T=11, BF16 bucketing cut Hot-8 from 136.2 to 78.8 us and led the mean from T=12.
// At T=17, FP4 MMA beat short FP4, 146.7 vs 148.3 us.
// At T=20, 64-column TMA beat FP4 MMA, 154.3 vs 156.0 us, and led through T=128.
// At T=43, BF16 short16 overtook bucketing, 303.2 vs 304.6 us.
// At T=192, BF16 MMA beat short16 (472.1 vs 478.8 us), and dense TMA beat 64-column
//        TMA (202.2 vs 208.9 us). Dense TMA reached 457.5 us at Uniform T=1024.
// At T=4096, BF16 bucketing recovered the lead over BF16 MMA, 3715.8 vs 3782.9 us.
constexpr std::array choices{
    TokenChoice{1, Schedule::VectorComplete, Schedule::VectorComplete},
    TokenChoice{2, Schedule::VectorDecodeSplit, Schedule::DecodeA4Reuse},
    TokenChoice{9, Schedule::VectorDecodeSplit, Schedule::Fp4Short},
    TokenChoice{11, Schedule::Bf16Bucketed, Schedule::Fp4Short},
    TokenChoice{17, Schedule::Bf16Bucketed, Schedule::Fp4Mma},
    TokenChoice{20, Schedule::Bf16Bucketed, Schedule::Sm120Tma64},
    TokenChoice{43, Schedule::Bf16Short16, Schedule::Sm120Tma64},
    TokenChoice{192, Schedule::Bf16Mma, Schedule::Sm120TmaDense},
    TokenChoice{4096, Schedule::Bf16Bucketed, Schedule::Sm120TmaDense},
};

constexpr auto allocation_boundaries = [] {
    std::array<int, choices.size()> boundaries{};
    for (std::size_t i = 1; i < choices.size(); ++i) boundaries[i - 1] = choices[i].first;
    boundaries.back() = softcap::single_block_tokens + 1;
    return boundaries;
}();

Schedule measured_schedule(int tokens, bool allow_a4) {
    for (auto it = choices.rbegin(); it != choices.rend(); ++it)
        if (tokens >= it->first) return allow_a4 ? it->a4 : it->a16;
    throw std::logic_error("routed SwiGLU: invalid slice extent");
}

Plan select_plan(const SwiGLUProfile& profile, int tokens, int) {
    const bool allow_a4 = allows_a4(profile.gate_up_policy) && allows_a4(profile.down_policy);
    const auto schedule = measured_schedule(tokens, allow_a4);
    int columns         = 32;
    switch (schedule) {
    case Schedule::Fp4Short:
        columns = 8;
        break;
    case Schedule::Bf16Short16:
        columns = 16;
        break;
    case Schedule::Bf16Bucketed:
    case Schedule::Sm120Tma64:
        columns = 64;
        break;
    case Schedule::Sm120TmaDense:
        columns = 128;
        break;
    default:
        break;
    }
    Plan plan;
    plan.implementation = Implementation::SoftcapNvfp4;
    plan.softcap        = {.schedule              = schedule,
                           .columns               = columns,
                           .single_block_grouping = tokens <= softcap::single_block_tokens};
    return plan;
}
} // namespace

const ShapePolicy& nvfp4_mul_h2304_i1152_e128_k8() {
    static const ShapePolicy shape{.select                = select_plan,
                                   .allocation_boundaries = allocation_boundaries,
                                   .all_slice_tokens      = softcap::slice_tokens,
                                   .masked_slice_tokens   = softcap::slice_tokens,
                                   .scaled_shared         = false};
    return shape;
}
} // namespace ninfer::ops::detail::moe_routed
