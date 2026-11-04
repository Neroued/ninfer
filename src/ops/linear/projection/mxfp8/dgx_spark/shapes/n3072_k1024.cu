#include "ops/linear/projection/mxfp8/launch.cuh"
#include <array>
#include <limits>

namespace ninfer::ops::detail {
namespace {
template <class S>
using Static = NativeFp8ScheduleInstance<S, 1024>;
constexpr std::array<Mxfp8LinearRoute, 2> a16{{
    {32, mxfp8_linear_sliced<Static<NativeFp8A16SlicedKSchedule<32, 8, 4, 128, 2>>>,
     Mxfp8LinearCompute::A16},
    {std::numeric_limits<int>::max(),
     mxfp8_linear_mma<Static<NativeFp8MmaSchedule<64, 128, 128, 32, 32, 2>>>,
     Mxfp8LinearCompute::A16},
}};
constexpr std::array<Mxfp8LinearRoute, 4> a8{{
    {32, mxfp8_linear_sliced<Static<NativeFp8A16SlicedKSchedule<32, 8, 4, 128, 2>>>,
     Mxfp8LinearCompute::A16},
    {64, mxfp8_linear_a8<Static<NativeFp8MmaSchedule<64, 32, 128, 32, 16, 4>>>,
     Mxfp8LinearCompute::A8},
    {128, mxfp8_linear_a8<Static<NativeFp8MmaSchedule<64, 32, 128, 32, 16, 3>>>,
     Mxfp8LinearCompute::A8},
    {std::numeric_limits<int>::max(),
     mxfp8_linear_tma<Static<Mxfp8A8TmaSchedule<128, 128, 128, 64, 32, 2>>, true>,
     Mxfp8LinearCompute::A8},
}};
} // namespace

const Mxfp8LinearShape kMxfp8N3072K1024{3072, 1024, a16, a8};
} // namespace ninfer::ops::detail
