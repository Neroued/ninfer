#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using S0 = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<4, 2, 1, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    2560>;
using S1 = Bf16ScheduleInstance<
    Bf16A16SimtSchedule<4, 2, 1, 8, 4, 2, Bf16SimtActivationAccess::WarpPacked,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 2, 2>,
    2560>;
using S2 = Bf16A16SimtSchedule<4, 2, 1, 8, 4, 2, Bf16SimtActivationAccess::DirectStream,
                               Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 2, 2>;
using S3 = Bf16A16SimtSchedule<8, 2, 1, 8, 4, 2, Bf16SimtActivationAccess::WarpPacked,
                               Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 2, 2>;
using S4 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 8, 64, 2>, 2560>;
using S5 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 32, 8, 64, 2>, 2560>;
} // namespace

Bf16Launch select_bf16_n48_k2560(std::int32_t tokens) {
    if (tokens <= 1) return launch_bf16_gemv<S0>;
    if (tokens <= 8) return launch_bf16_simt<S1>;
    if (tokens <= 12) return launch_bf16_simt<S2>;
    if (tokens <= 16) return launch_bf16_simt<S3>;
    if (tokens <= 128) return launch_bf16_sliced_k_mma<S4>;
    return launch_bf16_sliced_k_mma<S5>;
}
} // namespace ninfer::ops::detail
