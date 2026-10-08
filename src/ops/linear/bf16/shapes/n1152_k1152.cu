#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
using S0 = Bf16KTailSchedule<Bf16ScheduleInstance<
    Bf16A16SimtSchedule<8, 4, 1, 8, 4, 4, Bf16SimtActivationAccess::DirectStream,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 2, 4>,
    1152>>;
using S1 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 3, 128, 2>, 1152>;
using S2 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<32, 8, 3, 128, 2>, 1152>;
using S3 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<32, 16, 3, 128, 2>, 1152>;
using S4 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<32, 32, 3, 64, 2>, 1152>;
using S5 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 32, 64, 16, 8, 3, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    1152>;
using S6 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<32, 64, 64, 16, 8, 3, 1>, 1152>;
using S7 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>, 1152>;
using S8 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 3, 1>, 1152>;
using S9 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, 1152>;
} // namespace

Bf16Launch select_bf16_n1152_k1152(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0)
        throw std::invalid_argument("bf16 linear: P must be a multiple of 4 in [4,131072]");
    if (tokens <= 4) return launch_bf16_simt<S0>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<S1>;
    if (tokens <= 32) return launch_bf16_sliced_k_mma<S2>;
    if (tokens <= 40) return launch_bf16_sliced_k_mma<S3>;
    if (tokens <= 64) return launch_bf16_sliced_k_mma<S4>;
    if (tokens <= 128) return launch_bf16_mma<S5>;
    if (tokens <= 256) return launch_bf16_tma_mma<S6>;
    if (tokens <= 512) return launch_bf16_tma_mma<S7>;
    if (tokens <= 1024) return launch_bf16_tma_mma<S8>;
    if (tokens <= 2048) return launch_bf16_tma_mma<S9>;
    return launch_bf16_tma_mma<S7>;
}
} // namespace ninfer::ops::detail
