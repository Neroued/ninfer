#include "ops/linear/projection/fp8_block/launch.cuh"
#include <array>
#include <limits>

namespace ninfer::ops::detail {
namespace {
template <class S>
using Static = NativeFp8ScheduleInstance<S, 1024>;

template <class S>
struct RegisterWeights : S {
    static constexpr bool kRegisterWeights = true;
};

Fp8BlockLinearRoute select(int tokens, LinearPolicy) {
    if (tokens <= 2)
        return {fp8_block_linear_simt<Static<NativeFp8A16SimtSchedule<2, 8, 1, 16, 4, 1, true>>>,
                false};
    if (tokens <= 8)
        return {fp8_block_linear_sliced<Static<NativeFp8A16SlicedKSchedule<16, 8, 8, 128, 1>>>,
                true};
    if (tokens <= 16)
        return {fp8_block_linear_sliced<Static<NativeFp8A16SlicedKSchedule<16, 8, 4, 128, 1>>>,
                true};
    if (tokens <= 32)
        return {fp8_block_linear_sliced<Static<NativeFp8A16SlicedKSchedule<16, 16, 4, 128, 1>>>,
                true};
    if (tokens <= 64)
        return {
            fp8_block_linear_mma<
                RegisterWeights<Static<NativeFp8MmaSchedule<32, 32, 128, 16, 16, 2, 2, 2>>>, true>,
            true};
    if (tokens <= 96)
        return {
            fp8_block_linear_mma<
                RegisterWeights<Static<NativeFp8MmaSchedule<64, 32, 128, 32, 16, 2, 2, 2>>>, true>,
            true};
    if (tokens <= 128)
        return {fp8_block_linear_mma<
                    RegisterWeights<Static<NativeFp8MmaSchedule<64, 64, 128, 16, 32, 2, 2>>>>,
                true};
    if (tokens <= 256)
        return {fp8_block_linear_tma<Static<Fp8BlockA16TmaSchedule<128, 64, 128, 32, 32, 2, 8>>>,
                true};
    return {fp8_block_linear_tma<Static<Fp8BlockA16TmaSchedule<128, 128, 64, 16, 128, 3, 8>>>,
            true};
}
} // namespace

const Fp8BlockLinearShape kFp8BlockN4096K1024{4096, 1024, select};
} // namespace ninfer::ops::detail
