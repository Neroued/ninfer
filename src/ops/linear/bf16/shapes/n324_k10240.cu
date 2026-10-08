#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using S0 = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<2, 1, 1, 16, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    10240>;
using S1 = Bf16ScheduleInstance<
    Bf16A16SimtSchedule<8, 2, 1, 8, 4, 2, Bf16SimtActivationAccess::DirectStream,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 2, 2>,
    10240>;
using S2 =
    Bf16RowTailSchedule<Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 4, 128, 3>, 10240>>;
using S3 =
    Bf16RowTailSchedule<Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 16, 4, 128, 3>, 10240>>;
using S4 =
    Bf16RowTailSchedule<Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 16, 4, 64, 6>, 10240>>;
using S5 = Bf16RowTailSchedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<32, 64, 64, 16, 8, 4, 1>, 10240>>;
using S6 = Bf16RowTailSchedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 48, 64, 16, 8, 4, 1>, 10240>>;
} // namespace

Bf16Launch select_bf16_n324_k10240(std::int32_t tokens) {
    if (tokens <= 1) return launch_bf16_gemv<S0>;
    if (tokens <= 8) return launch_bf16_simt<S1>;
    if (tokens <= 48) return launch_bf16_sliced_k_mma<S2>;
    if (tokens <= 96) return launch_bf16_sliced_k_mma<S3>;
    if (tokens <= 128) return launch_bf16_sliced_k_mma<S4>;
    if (tokens <= 512) return launch_bf16_tma_mma<S5>;
    return launch_bf16_tma_mma<S6>;
}
} // namespace ninfer::ops::detail
