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
    {4, mxfp8_swiglu_simt<Static<PairWarps<NativeFp8A16SimtSchedule<4, 4, 1, 16>>>>, false, 1,
     false},
    {32, mxfp8_swiglu_sliced<Static<NativeFp8A16SlicedKSchedule<16, 16, 4, 128, 2>>>, false},
    {128,
     mxfp8_swiglu_a16<Static<RegisterWeights<NativeFp8MmaSchedule<64, 64, 128, 16, 32, 2, 2>>>>,
     false},
    {std::numeric_limits<int>::max(),
     mxfp8_swiglu_a16<Static<RegisterWeights<NativeFp8MmaSchedule<128, 64, 128, 32, 32, 1, 2>>>>,
     false},
}};

template <class S, template <class> class Epilogue = BasicSwiGluInterleavedMmaEpilogue,
          class Rows = SwiGluInterleavedRows<8>>
constexpr Mxfp8SwiGluRoute padded(int last) {
    return {last, mxfp8_swiglu_padded_tma<S, 6144, Epilogue, Rows>, true, S::kBlockTokens, true,
            true};
}

using Front              = Static<NativeFp8MmaSchedule<64, 64, 128, 32, 16, 3>>;
using Transition         = Static<Mxfp8A8TmaSchedule<64, 128, 128, 32, 32, 3>>;
using Near               = Static<Mxfp8A8TmaSchedule<128, 64, 128, 64, 16, 3, 1, 2, true>>;
using Mid                = Static<Mxfp8A8TmaSchedule<192, 128, 128, 96, 32, 2, 1, 8>>;
using Bulk               = Static<Mxfp8A8TmaSchedule<192, 128, 128, 48, 64, 2, 1, 8, false, 1>>;
using Large              = Static<Mxfp8A8TmaSchedule<192, 128, 128, 48, 64, 2, 1, 2, false, 1>>;
constexpr int kAllTokens = std::numeric_limits<int>::max();
constexpr std::array<Mxfp8SwiGluRoute, 14> a8{{
    {1, mxfp8_swiglu_sliced<Static<NativeFp8A16SlicedKSchedule<8, 4, 8, 256, 2>>>, false},
    {2, mxfp8_swiglu_sliced<Static<NativeFp8A16SlicedKSchedule<8, 1, 16, 128, 2>>>, false},
    {4, mxfp8_swiglu_sliced<Static<NativeFp8A16SlicedKSchedule<8, 2, 8, 256, 2>>>, false},
    {8, mxfp8_swiglu_sliced<Static<NativeFp8A16SlicedKSchedule<8, 4, 8, 256, 1>>>, false},
    {16, mxfp8_swiglu_sliced<Static<NativeFp8A16SlicedKSchedule<16, 8, 8, 128, 2>>>, false},
    {32, mxfp8_swiglu_sliced<Static<NativeFp8A16SlicedKSchedule<16, 16, 4, 128, 2>>>, false},
    {48, mxfp8_swiglu_sliced<Static<NativeFp8A16SlicedKSchedule<32, 32, 4, 128, 1>>>, false},
    {128, mxfp8_swiglu_padded_mma<Front, 6144>, true, Front::kBlockTokens, true, true},
    padded<Transition>(256),
    padded<Near, BasicSwiGluRowMajorBf16VectorEpilogue, SwiGluTmaBranchRows<64>>(320),
    {384, mxfp8_swiglu_padded_tma<Mid, 6144>, true, Mid::kBlockTokens, true, true},
    padded<Bulk>(2048),
    padded<Large>(kAllTokens / Large::kBlockTokens * Large::kBlockTokens),
    {kAllTokens, mxfp8_swiglu_tma<Large>, true},
}};
} // namespace

const Mxfp8SwiGluShape kMxfp8SwiGluN12288K4096{12288, 4096, a16, a8};
} // namespace ninfer::ops::detail
