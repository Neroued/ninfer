#include "ops/linear/projection/mxfp8/launch.cuh"
#include <array>
#include <limits>

namespace ninfer::ops::detail {
namespace {
template <class S>
using Static = NativeFp8ScheduleInstance<S, 1024>;
constexpr std::array<Mxfp8LinearRoute, 7> a16{{
    {2, mxfp8_linear_sliced<Static<NativeFp8A16SlicedKSchedule<32, 8, 4, 128, 2>>>,
     Mxfp8LinearCompute::A16},
    {4, mxfp8_linear_sliced<Static<NativeFp8A16SlicedKSchedule<16, 8, 8, 128, 1>>>,
     Mxfp8LinearCompute::A16},
    {8, mxfp8_linear_sliced<Static<NativeFp8A16SlicedKSchedule<32, 8, 4, 128, 2>>>,
     Mxfp8LinearCompute::A16},
    {32, mxfp8_linear_sliced<Static<NativeFp8A16SlicedKSchedule<16, 16, 2, 128, 2>>>,
     Mxfp8LinearCompute::A16},
    {256, mxfp8_linear_mma<Static<NativeFp8MmaSchedule<32, 32, 128, 16, 16, 2>>>,
     Mxfp8LinearCompute::A16},
    {512, mxfp8_linear_mma<Static<NativeFp8MmaSchedule<64, 64, 128, 32, 16, 2>>>,
     Mxfp8LinearCompute::A16},
    {std::numeric_limits<int>::max(),
     mxfp8_linear_mma<Static<NativeFp8MmaSchedule<64, 128, 128, 32, 32, 2>>>,
     Mxfp8LinearCompute::A16},
}};
constexpr std::array<Mxfp8LinearRoute, 8> a8{{
    {8, mxfp8_linear_sliced<Static<NativeFp8A16SlicedKSchedule<32, 8, 4, 128, 2>>>,
     Mxfp8LinearCompute::A16},
    {32, mxfp8_linear_a8<Static<NativeFp8MmaSchedule<32, 8, 128, 16, 8, 2>>>,
     Mxfp8LinearCompute::A8},
    {96, mxfp8_linear_a8<Static<NativeFp8MmaSchedule<64, 16, 128, 32, 8, 3>>>,
     Mxfp8LinearCompute::A8},
    {256, mxfp8_linear_a8<Static<NativeFp8MmaSchedule<64, 32, 128, 32, 16, 2>>>,
     Mxfp8LinearCompute::A8},
    {384, mxfp8_linear_tma<Static<Mxfp8A8TmaSchedule<128, 64, 128, 64, 16, 3, 1, 2, true, 2>>, true>,
     Mxfp8LinearCompute::A8},
    {768, mxfp8_linear_tma<Static<Mxfp8A8TmaSchedule<128, 128, 128, 64, 64, 2>>, true>,
     Mxfp8LinearCompute::A8},
    {1024, mxfp8_linear_tma<Static<Mxfp8A8TmaSchedule<192, 128, 128, 48, 64, 2, 1, 2, true, 1>>, true>,
     Mxfp8LinearCompute::A8},
    {std::numeric_limits<int>::max(),
     mxfp8_linear_tma_vector<Static<Mxfp8A8TmaSchedule<128, 128, 128, 64, 32, 2, 1, 8, false>>>,
     Mxfp8LinearCompute::A8},
}};
} // namespace

const Mxfp8LinearShape kMxfp8N1024K1024{1024, 1024, a16, a8};
} // namespace ninfer::ops::detail
