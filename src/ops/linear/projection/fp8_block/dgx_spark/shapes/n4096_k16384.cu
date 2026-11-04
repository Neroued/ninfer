#include "ops/linear/projection/fp8_block/launch.cuh"
#include <array>
#include <limits>

namespace ninfer::ops::detail {
namespace {
template <class S>
using Static = NativeFp8ScheduleInstance<S, 16384>;

template <class S>
struct RegisterWeights : S {
    static constexpr bool kRegisterWeights = true;
};

Fp8BlockLinearRoute select(int tokens, LinearPolicy) {
    if (tokens <= 4)
        return {fp8_block_linear_simt<Static<NativeFp8A16SimtSchedule<4, 8, 1, 16>>>, false};
    if (tokens <= 16)
        return {fp8_block_linear_sliced<Static<NativeFp8A16SlicedKSchedule<16, 8, 4, 128, 2>>>,
                true};
    if (tokens <= 32)
        return {fp8_block_linear_sliced<Static<NativeFp8A16SlicedKSchedule<16, 16, 4, 128, 2>>>,
                true};
    if (tokens <= 64 || (tokens > 96 && tokens <= 128))
        return {fp8_block_linear_mma<
                    RegisterWeights<Static<NativeFp8MmaSchedule<64, 64, 128, 16, 32, 2, 2>>>>,
                true};
    if (tokens <= 96)
        return {fp8_block_linear_mma<Static<NativeFp8MmaSchedule<64, 32, 128, 16, 16, 2>>>, true};
    return {fp8_block_linear_mma<
                RegisterWeights<Static<NativeFp8MmaSchedule<128, 64, 128, 32, 32, 1, 2>>>, true>,
            true};
}
} // namespace

const Fp8BlockLinearShape kFp8BlockN4096K16384{4096, 16384, select};
} // namespace ninfer::ops::detail
