#include "ops/linear/swiglu/mxfp8/dispatch.h"
#include "ops/linear/swiglu/mxfp8/launch.cuh"
#include <array>
#include <limits>

namespace ninfer::ops::detail {
namespace {
template <class S>
using Static = NativeFp8ScheduleInstance<S, 4096>;

template <class S>
struct PairWarps : S {
    static constexpr bool kPairAcrossWarps = true;
};

template <class S>
struct RegisterWeights : S {
    static constexpr bool kRegisterWeights = true;
};

constexpr std::array<Mxfp8SwiGluRoute, 4> a16{{
    {2, mxfp8_swiglu_simt<Static<PairWarps<NativeFp8A16SimtSchedule<2, 4, 1, 16>>>>, false, 1,
     false},
    {16, mxfp8_swiglu_sliced<Static<NativeFp8A16SlicedKSchedule<16, 8, 8, 128, 1>>>, false},
    {1024,
     mxfp8_swiglu_a16<Static<RegisterWeights<NativeFp8MmaSchedule<64, 64, 128, 16, 32, 2, 2>>>>,
     false},
    {std::numeric_limits<int>::max(),
     mxfp8_swiglu_a16<Static<RegisterWeights<NativeFp8MmaSchedule<128, 64, 128, 32, 32, 1, 2>>>>,
     false},
}};
constexpr std::array<Mxfp8SwiGluRoute, 5> a8{{
    {2, mxfp8_swiglu_simt<Static<PairWarps<NativeFp8A16SimtSchedule<2, 4, 1, 16>>>>, false, 1,
     false},
    {16, mxfp8_swiglu_sliced<Static<NativeFp8A16SlicedKSchedule<16, 8, 8, 128, 1>>>, false},
    {64, mxfp8_swiglu_a8<Static<NativeFp8MmaSchedule<64, 32, 128, 32, 16, 3>>>, true},
    {512, mxfp8_swiglu_tma<Static<Mxfp8A8TmaSchedule<64, 128, 128, 32, 32, 3>>>, true},
    {std::numeric_limits<int>::max(),
     mxfp8_swiglu_tma<Static<Mxfp8A8TmaSchedule<128, 128, 128, 64, 32, 2>>>, true},
}};
} // namespace

const Mxfp8SwiGluShape kMxfp8SwiGluN2048K4096{2048, 4096, a16, a8};
} // namespace ninfer::ops::detail
