#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
using S0 = Bf16RowTailSchedule<Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 16, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    1152>>;
using S1 = Bf16RowTailSchedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 32, 16, 3, 2>, 1152>>;
using S2 = Bf16RowTailSchedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>, 1152>>;
using S3 = Bf16RowTailSchedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 2, 2>, 1152>>;
using S4 = Bf16RowTailSchedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, 1152>>;
using S5 = Bf16RowTailSchedule<Bf16A16TmaMmaSchedule<96, 64, 64, 48, 32, 2, 2>>;
using S6 = Bf16RowTailSchedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 128, 64, 64, 32, 2, 2>, 1152>>;
} // namespace

Bf16Launch select_bf16_n4304_k1152(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0)
        throw std::invalid_argument("bf16 linear: P must be a multiple of 4 in [4,131072]");
    if (tokens <= 32) return launch_bf16_mma<S0>;
    if (tokens <= 96) return launch_bf16_tma_mma<S1>;
    if (tokens <= 128) return launch_bf16_tma_mma<S2>;
    if (tokens <= 256) return launch_bf16_tma_mma<S3>;
    if (tokens <= 512) return launch_bf16_tma_mma<S4>;
    if (tokens <= 1024) return launch_bf16_tma_mma<S2>;
    if (tokens <= 2048) return launch_bf16_tma_mma<S5>;
    return launch_bf16_tma_mma<S6>;
}
} // namespace ninfer::ops::detail
