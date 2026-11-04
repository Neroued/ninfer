#include "ops/linear/projection/mxfp8/launch.cuh"
#include <array>
#include <limits>

namespace ninfer::ops::detail {
namespace {
template <class S>
using Static = NativeFp8ScheduleInstance<S, 1024>;
constexpr std::array<Mxfp8LinearRoute, 5> a16{{
    {1, mxfp8_linear_simt<Static<NativeFp8A16SimtSchedule<1, 8, 1, 16, 4, 1, true>>>,
     Mxfp8LinearCompute::Simt},
    {32, mxfp8_linear_sliced<Static<NativeFp8A16SlicedKSchedule<16, 16, 4, 128, 2>>>,
     Mxfp8LinearCompute::A16},
    {64, mxfp8_linear_mma<Static<NativeFp8MmaSchedule<32, 32, 128, 16, 16, 2>>>,
     Mxfp8LinearCompute::A16},
    {128, mxfp8_linear_mma<Static<NativeFp8MmaSchedule<64, 64, 128, 32, 16, 2>>>,
     Mxfp8LinearCompute::A16},
    {std::numeric_limits<int>::max(),
     mxfp8_linear_mma<Static<NativeFp8MmaSchedule<64, 128, 128, 32, 32, 2>>>,
     Mxfp8LinearCompute::A16},
}};
constexpr std::array<Mxfp8LinearRoute, 10> a8{{
    {1, mxfp8_linear_simt<Static<NativeFp8A16SimtSchedule<1, 8, 1, 16, 4, 1, true>>>,
     Mxfp8LinearCompute::Simt},
    {4, mxfp8_linear_sliced<Static<NativeFp8A16SlicedKSchedule<8, 2, 8, 128, 1>>>,
     Mxfp8LinearCompute::A16},
    {8, mxfp8_linear_sliced<Static<NativeFp8A16SlicedKSchedule<8, 4, 8, 128, 1>>>,
     Mxfp8LinearCompute::A16},
    {16, mxfp8_linear_sliced<Static<NativeFp8A16SlicedKSchedule<16, 8, 4, 128, 2>>>,
     Mxfp8LinearCompute::A16},
    {32, mxfp8_linear_a8<Static<NativeFp8MmaSchedule<64, 16, 128, 32, 8, 3>>>,
     Mxfp8LinearCompute::A8},
    {128, mxfp8_linear_a8<Static<NativeFp8MmaSchedule<64, 64, 128, 32, 16, 2>>>,
     Mxfp8LinearCompute::A8},
    {256, mxfp8_linear_tma<Static<Mxfp8A8TmaSchedule<64, 64, 128, 32, 16, 2, 1, 2>>, true>,
     Mxfp8LinearCompute::A8},
    {384, mxfp8_linear_tma<Static<Mxfp8A8TmaSchedule<128, 128, 128, 64, 64, 2>>, true>,
     Mxfp8LinearCompute::A8},
    {512,
     mxfp8_linear_tma<Static<Mxfp8A8TmaSchedule<192, 128, 128, 48, 64, 2, 1, 2, true, 1>>, true>,
     Mxfp8LinearCompute::A8},
    {std::numeric_limits<int>::max(),
     mxfp8_linear_tma_vector<Static<Mxfp8A8TmaSchedule<128, 128, 128, 64, 64, 2>>>,
     Mxfp8LinearCompute::A8},
}};
} // namespace

const Mxfp8LinearShape kMxfp8N4096K1024{4096, 1024, a16, a8};
} // namespace ninfer::ops::detail
