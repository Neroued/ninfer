#include "ops/linear/projection/mxfp8/launch.cuh"
#include <array>
#include <limits>

namespace ninfer::ops::detail {
namespace {
template <class S>
using Static = NativeFp8ScheduleInstance<S, 4096>;
constexpr std::array<Mxfp8LinearRoute, 4> a16{{
    {8, mxfp8_linear_sliced<Static<NativeFp8A16SlicedKSchedule<16, 8, 4, 128, 2>>>,
     Mxfp8LinearCompute::A16},
    {32, mxfp8_linear_sliced<Static<NativeFp8A16SlicedKSchedule<32, 8, 4, 128, 2>>>,
     Mxfp8LinearCompute::A16},
    {64, mxfp8_linear_mma<Static<NativeFp8MmaSchedule<32, 32, 128, 16, 16, 2>>>,
     Mxfp8LinearCompute::A16},
    {std::numeric_limits<int>::max(),
     mxfp8_linear_mma<Static<NativeFp8MmaSchedule<64, 128, 128, 32, 32, 2>>>,
     Mxfp8LinearCompute::A16},
}};
constexpr std::array<Mxfp8LinearRoute, 10> a8{{
    {16, mxfp8_linear_sliced<Static<NativeFp8A16SlicedKSchedule<16, 8, 4, 128, 2>>>,
     Mxfp8LinearCompute::A16},
    {32, mxfp8_linear_sliced<Static<NativeFp8A16SlicedKSchedule<16, 16, 4, 128, 2>>>,
     Mxfp8LinearCompute::A16},
    {40, mxfp8_linear_sliced<Static<NativeFp8A16SlicedKSchedule<32, 8, 4, 128, 2>>>,
     Mxfp8LinearCompute::A16},
    {128, mxfp8_linear_tma<Static<Mxfp8A8TmaSchedule<48, 64, 256, 16, 32, 2, 1, 2>>, true>,
     Mxfp8LinearCompute::A8},
    {256, mxfp8_linear_tma<Static<Mxfp8A8TmaSchedule<128, 128, 128, 64, 64, 2>>, true>,
     Mxfp8LinearCompute::A8},
    {320, mxfp8_linear_tma<Static<Mxfp8A8TmaSchedule<128, 64, 128, 64, 16, 3, 1, 2, true, 2>>, true>,
     Mxfp8LinearCompute::A8},
    {512, mxfp8_linear_tma<Static<Mxfp8A8TmaSchedule<192, 128, 128, 48, 64, 2, 1, 2, true, 1>>, true>,
     Mxfp8LinearCompute::A8},
    {640, mxfp8_linear_tma<Static<Mxfp8A8TmaSchedule<128, 128, 128, 64, 32, 2, 1, 2, true, 2>>, true>,
     Mxfp8LinearCompute::A8},
    {704, mxfp8_linear_tma<Static<Mxfp8A8TmaSchedule<128, 64, 128, 64, 16, 3, 1, 2, true, 2>>, true>,
     Mxfp8LinearCompute::A8},
    {std::numeric_limits<int>::max(),
     mxfp8_linear_tma<Static<Mxfp8A8TmaSchedule<192, 128, 128, 48, 64, 2, 1, 2, true, 1>>, true>,
     Mxfp8LinearCompute::A8},
}};
} // namespace

const Mxfp8LinearShape kMxfp8N2176K4096{2176, 4096, a16, a8};
} // namespace ninfer::ops::detail
