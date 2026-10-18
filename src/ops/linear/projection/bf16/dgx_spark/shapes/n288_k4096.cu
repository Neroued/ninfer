#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Schedule>
using Static = Bf16ScheduleInstance<Schedule, 4096>;
template <class Schedule>
using Tail = Static<Bf16KTailSchedule<Schedule>>;

using Gemv =
    Tail<Bf16A16GemvSchedule<4, 4, 1, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                             Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>;
using Short      = Tail<Bf16A16SlicedKMmaSchedule<16, 8, 16, 64, 2>>;
using Hot        = Tail<Bf16A16SlicedKMmaSchedule<16, 16, 8, 64, 2>>;
using Transition = Tail<Bf16A16SlicedKMmaSchedule<16, 16, 4, 64, 2>>;
using Medium     = Tail<Bf16A16SlicedKMmaSchedule<16, 32, 4, 64, 2>>;
using Mma16 =
    Static<Bf16A16MmaSchedule<48, 16, 128, 16, 8, 3, 1, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Mma32 =
    Static<Bf16A16MmaSchedule<48, 32, 256, 16, 8, 2, 1, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Tma32 = Static<Bf16A16TmaMmaSchedule<48, 32, 128, 16, 16, 2, 2, Bf16MmaRaster::RowFast>>;
using Tma64 = Static<Bf16A16TmaMmaSchedule<48, 64, 64, 16, 32, 2, 2, Bf16MmaRaster::RowFast>>;
} // namespace

Bf16Launch select_bf16_n288_k4096(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Gemv>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<Short>;
    if (tokens <= 32) return launch_bf16_sliced_k_mma<Hot>;
    if (tokens <= 40) return launch_bf16_sliced_k_mma<Transition>;
    if (tokens <= 64) return launch_bf16_sliced_k_mma<Medium>;
    if (tokens <= 128) return launch_bf16_mma<Mma16>;
    if (tokens <= 256) return launch_bf16_mma<Mma32>;
    if (tokens <= 512) return launch_bf16_tma_mma<Tma32>;
    return launch_bf16_tma_mma<Tma64>;
}
} // namespace ninfer::ops::detail
