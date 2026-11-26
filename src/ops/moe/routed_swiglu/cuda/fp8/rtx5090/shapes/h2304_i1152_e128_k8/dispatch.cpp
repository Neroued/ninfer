#include "ops/moe/routed_swiglu/cuda/dispatch.h"

#include <array>

namespace ninfer::ops::detail::moe_routed {
namespace {
using mxfp8_softcap::Schedule;

struct TokenChoice {
    int first;
    Schedule bf16, native;
};

// RTX 5090 Graph body. Geometric means use all five routing distributions.
// The dispatch is independent of cache residency. The cold sweep established the
// schedule ranges, and the cold/warm races below compare adjacent schedules at the
// same token count. Times are in us.
//
// At T=3, vector beat BF16-16 in both cache states (95.3/88.9 vs 97.3/101.1,
// cold/warm). Against native-16 it led on Uniform (117.8/124.0 vs 124.0/130.9).
// At T=4, BF16-16 led vector (111.9/108.6 vs 116.0/116.6), and native-16
// improved further (107.8/103.3 vs vector's 116.0/111.0).
// At T=16, native-16 led native-32 (214.0/198.0 vs 216.2/203.1). At T=17 the
// order reversed (220.4/207.8 for native-32 vs 221.1/220.6 for native-16).
// At T=64, native-32 led native-128 on Uniform (635.7/644.2 vs 637.8/646.5),
// with the five-distribution mean close (266.6/265.2 vs 264.8/261.9).
// At T=65, native-128 led the mean (266.3/275.2 vs 274.7/302.6) and Hot-8
// (58.4/53.1 vs 66.6/90.6). BF16-32 still narrowly led BF16-64 at T=65
// in this check (300.5/319.5 vs 309.7/329.8).
// At T=1024, wide-128 led native-128 (498.5/508.3 vs 506.1/521.1).
// At T=2048, transposed Down was close to wide-128 (700.7/714.9 vs 701.9/710.2).
//
// Cold sweep evidence for the surrounding ranges:
// T=1 measured 46.1 us for vector and 52.2 us for native-16. Over T=2..3,
// vector averaged 83.1 us versus 84.7 us for the fastest native-16 variant.
// After removing the adaptive split, native-16 averaged 129.9 us over T=4..8
// versus 132.5 us for grouped BF16-16. At T=8, both grouped paths measured
// about 52.2 us on Hot-8, versus 66.4 us for the previous adaptive schedule.
// Over T=17..32, native-32 beat native-16 in both halves of the interval.
// At T=64 on Hot-8 it measured 58.7 us versus 72.7 us for native-16.
// At T=512 the narrower row tiles still led on Uniform (660.5 vs 674.9 us).
// At T=1024 the wider tiles improved the five-distribution mean (504.1 to
// 498.8 us) and held Uniform at 704.4 us and Hot-8 at 273.4 us. The
// transposed Down tile led the mean at T=2048 and 4096 (697.7 and 1144.9 us
// versus 698.0 and 1151.5 us); Uniform T=4096 was 1205.3 vs 1214.8 us.
constexpr std::array choices{
    TokenChoice{1, Schedule::Vector, Schedule::Vector},
    TokenChoice{4, Schedule::Grouped16, Schedule::Grouped16},
    TokenChoice{17, Schedule::Grouped32, Schedule::Grouped32},
    TokenChoice{65, Schedule::Grouped64, Schedule::Grouped128},
    TokenChoice{1024, Schedule::Grouped64, Schedule::Wide128},
    TokenChoice{2048, Schedule::Grouped64, Schedule::Wide128Transposed},
};

constexpr auto allocation_boundaries = [] {
    std::array<int, choices.size()> out{};
    for (std::size_t i = 1; i < choices.size(); ++i) out[i - 1] = choices[i].first;
    out.back() = mxfp8_softcap::single_block_tokens + 1;
    return out;
}();

Plan select_plan(const SwiGLUProfile& profile, int tokens, int) {
    const bool gate_a8 = allows_a8(profile.gate_up_policy);
    const bool down_a8 = allows_a8(profile.down_policy);
    // Mixed policies use the BF16 tile widths with stage-specific arithmetic.
    auto choice = choices.rbegin();
    while (tokens < choice->first) ++choice;
    const auto schedule = gate_a8 && down_a8 ? choice->native : choice->bf16;
    Plan out;
    out.implementation = Implementation::SoftcapMxfp8;
    out.mxfp8          = {.schedule     = schedule,
                          .gate_a8      = schedule != Schedule::Vector && gate_a8,
                          .down_a8      = schedule != Schedule::Vector && down_a8,
                          .single_block = tokens <= mxfp8_softcap::single_block_tokens};
    return out;
}
} // namespace

const ShapePolicy& mxfp8_h2304_i1152_e128_k8() {
    static const ShapePolicy policy{.select                = select_plan,
                                    .allocation_boundaries = allocation_boundaries,
                                    .all_slice_tokens      = mxfp8_softcap::slice_tokens,
                                    .masked_slice_tokens   = mxfp8_softcap::slice_tokens,
                                    .scaled_shared         = false};
    return policy;
}
} // namespace ninfer::ops::detail::moe_routed
