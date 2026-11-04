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
    if (tokens <= 16)
        return {fp8_block_linear_sliced<Static<NativeFp8A16SlicedKSchedule<16, 8, 4, 128, 2>>>,
                true};
    if (tokens <= 32)
        return {fp8_block_linear_sliced<Static<NativeFp8A16SlicedKSchedule<16, 16, 4, 128, 2>>>,
                true};
    if (tokens <= 64)
        return {fp8_block_linear_mma<Static<NativeFp8MmaSchedule<32, 32, 128, 16, 16, 2>>>, true};
    if (tokens <= 256)
        return {fp8_block_linear_mma<
                    RegisterWeights<Static<NativeFp8MmaSchedule<64, 64, 128, 16, 32, 2, 2>>>>,
                true};
    return {fp8_block_linear_mma<
                RegisterWeights<Static<NativeFp8MmaSchedule<128, 64, 128, 32, 32, 1, 2>>>>,
            true};
}
} // namespace

const Fp8BlockLinearShape kFp8BlockN2048K4096{2048, 4096, select};
} // namespace ninfer::ops::detail
