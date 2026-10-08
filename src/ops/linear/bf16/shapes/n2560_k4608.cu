#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
using S0 = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<8, 1, 1, 16, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Streaming,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    4608>;
using S1 = Bf16ScheduleInstance<
    Bf16A16SimtSchedule<1, 1, 1, 8, 4, 2, Bf16SimtActivationAccess::WarpPacked,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 2, 2>,
    4608>;
using S2 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 4, 128, 3>, 4608>;
using S3 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<64, 8, 4, 64, 2>, 4608>;
using S4 = Bf16RowTailSchedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<48, 32, 192, 16, 16, 3, 1>, 4608>>;
using S5 = Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<64, 32, 128, 16, 16, 3, 1, Bf16MmaRaster::TokenFast, 1, 18>, 4608>;
using S6 = Bf16RowTailSchedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<48, 64, 128, 16, 16, 3, 1>, 4608>>;
using S7 =
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 4, 1, Bf16MmaRaster::Grouped, 4>,
                         4608>;
using S8 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 3, 1>, 4608>;
using S9 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, 4608>;
} // namespace

Bf16Launch select_bf16_n2560_k4608(std::int32_t tokens) {
    if (tokens > 32768) throw std::invalid_argument("bf16 linear: V must be in [1,32768]");
    if (tokens <= 1) return launch_bf16_gemv<S0>;
    if (tokens <= 2) return launch_bf16_simt<S1>;
    if (tokens <= 8) return launch_bf16_sliced_k_mma<S2>;
    if (tokens <= 24) return launch_bf16_sliced_k_mma<S3>;
    if (tokens <= 64) return launch_bf16_tma_mma<S4>;
    if (tokens <= 96) return launch_bf16_tma_mma<S5>;
    if (tokens <= 128) return launch_bf16_tma_mma<S6>;
    if (tokens <= 256) return launch_bf16_tma_mma<S7>;
    if (tokens <= 512) return launch_bf16_tma_mma<S8>;
    return launch_bf16_tma_mma<S9>;
}
} // namespace ninfer::ops::detail
