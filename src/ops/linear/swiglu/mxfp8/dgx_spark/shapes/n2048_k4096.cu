#include "ops/linear/swiglu/mxfp8/dispatch.h"
#include "ops/linear/swiglu/mxfp8/launch.cuh"
#include <array>
#include <limits>

namespace ninfer::ops::detail {
namespace {
// Keep CUDA entry registration owned by this shape translation unit.
template <class S>
struct SwiGluN2048K4096Schedule : NativeFp8ScheduleInstance<S, 4096> {};

template <class S>
using Static = SwiGluN2048K4096Schedule<S>;

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

template <class S, template <class> class Epilogue = BasicSwiGluInterleavedMmaEpilogue,
          class Rows = SwiGluInterleavedRows<8>>
constexpr Mxfp8SwiGluRoute padded(int last) {
    return {last, mxfp8_swiglu_padded_tma<S, 1024, Epilogue, Rows>, true, S::kBlockTokens, true,
            true};
}

using Transition         = Static<Mxfp8A8TmaSchedule<64, 128, 128, 32, 32, 3>>;
using TransitionWide     = Static<Mxfp8A8TmaSchedule<64, 128, 128, 32, 32, 3, 1, 2>>;
using Front              = Static<NativeFp8MmaSchedule<64, 64, 128, 32, 16, 3, 1, 4>>;
using Bulk               = Static<Mxfp8A8TmaSchedule<128, 128, 128, 64, 32, 2, 1, 4, true, 1>>;
using Large              = Static<Mxfp8A8TmaSchedule<128, 128, 128, 64, 32, 2, 1, 1, true, 1>>;
constexpr int kAllTokens = std::numeric_limits<int>::max();
constexpr std::array<Mxfp8SwiGluRoute, 12> a8{{
    {4, mxfp8_swiglu_sliced<Static<NativeFp8A16SlicedKSchedule<8, 2, 8, 256, 2>>>, false},
    {16, mxfp8_swiglu_sliced<Static<NativeFp8A16SlicedKSchedule<16, 8, 8, 128, 1>>>, false},
    {24, mxfp8_swiglu_sliced<Static<NativeFp8A16SlicedKSchedule<16, 16, 4, 128, 2>>>, false},
    {48, mxfp8_swiglu_a8<Static<NativeFp8MmaSchedule<64, 32, 128, 32, 16, 3>>>, true},
    {64, mxfp8_swiglu_padded_mma<Front, 1024>, true, Front::kBlockTokens, true, true},
    padded<Transition>(128),
    {192, mxfp8_swiglu_padded_mma<Front, 1024>, true, Front::kBlockTokens, true, true},
    padded<Transition>(384),
    padded<TransitionWide>(512),
    padded<Bulk, BasicSwiGluRowMajorBf16VectorEpilogue, SwiGluTmaBranchRows<64>>(2048),
    padded<Large, BasicSwiGluRowMajorBf16VectorEpilogue, SwiGluTmaBranchRows<64>>(
        kAllTokens / Large::kBlockTokens * Large::kBlockTokens),
    {kAllTokens, mxfp8_swiglu_tma<Large>, true},
}};
} // namespace

const Mxfp8SwiGluShape kMxfp8SwiGluN2048K4096{2048, 4096, a16, a8};
} // namespace ninfer::ops::detail
