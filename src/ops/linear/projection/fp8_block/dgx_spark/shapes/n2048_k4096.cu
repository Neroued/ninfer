#include "ops/linear/projection/fp8_block/launch.cuh"
#include <array>
#include <limits>

namespace ninfer::ops::detail {
namespace {
template <class S>
using Static = NativeFp8ScheduleInstance<S, 4096>;

template <class S>
struct RegisterWeights : S {
    static constexpr bool kRegisterWeights = true;
};

Fp8BlockLinearRoute select(int tokens, LinearPolicy) {
    if (tokens <= 2)
        return {fp8_block_linear_simt<Static<NativeFp8A16SimtSchedule<2, 4, 1, 16, 4, 1, true>>>,
                false};
    if (tokens <= 16)
        return {fp8_block_linear_sliced<Static<NativeFp8A16SlicedKSchedule<16, 8, 4, 128, 2>>>,
                true};
    if (tokens <= 32)
        return {fp8_block_linear_sliced<Static<NativeFp8A16SlicedKSchedule<16, 16, 4, 128, 2>>>,
                true};
    if (tokens <= 64)
        return {fp8_block_linear_mma<
                    RegisterWeights<Static<NativeFp8MmaSchedule<48, 32, 128, 16, 16, 2, 2, 2>>>,
                    true>,
                true};
    if (tokens <= 128)
        return {fp8_block_linear_mma<
                    RegisterWeights<Static<NativeFp8MmaSchedule<96, 64, 128, 32, 16, 2, 1, 2>>>,
                    true>,
                true};
    if (tokens <= 192)
        return {fp8_block_linear_mma<
                    RegisterWeights<Static<NativeFp8MmaSchedule<128, 64, 128, 32, 32, 2, 1, 2>>>,
                    true>,
                true};
    if (tokens <= 256 || (tokens > 384 && tokens <= 640) ||
        (tokens > 768 && tokens <= 896) || (tokens > 960 && tokens <= 1024))
        return {fp8_block_linear_mma<
                    RegisterWeights<Static<NativeFp8MmaSchedule<64, 64, 128, 32, 32, 2, 2>>>,
                    true>,
                true};
    return {fp8_block_linear_mma<
                RegisterWeights<Static<NativeFp8MmaSchedule<128, 64, 128, 32, 32, 1, 2>>>>,
            true};
}
} // namespace

const Fp8BlockLinearShape kFp8BlockN2048K4096{2048, 4096, select};
} // namespace ninfer::ops::detail
