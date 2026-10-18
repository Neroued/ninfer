#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Schedule>
using Static = Bf16ScheduleInstance<Schedule, 1024>;
using Gemv =
    Static<Bf16A16GemvSchedule<4, 1, 1, 8, 4, Bf16ActivationAccess::Direct,
                               Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>;
using Short = Static<Bf16A16SlicedKMmaSchedule<16, 8, 4, 64, 2>>;
using Small = Static<Bf16A16SlicedKMmaSchedule<16, 16, 4, 64, 2>>;
using Medium =
    Static<Bf16A16MmaSchedule<64, 32, 64, 16, 16, 2, 2, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Large = Static<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 1>>;
using Bulk  = Static<Bf16A16TmaMmaSchedule<128, 128, 64, 32, 32, 3, 1>>;
} // namespace

Bf16Launch select_bf16_n4096_k1024(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Gemv>;
    if (tokens <= 8) return launch_bf16_sliced_k_mma<Short>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<Small>;
    if (tokens <= 128) return launch_bf16_mma<Medium>;
    if (tokens <= 256) return launch_bf16_tma_mma<Large>;
    return launch_bf16_tma_mma<Bulk>;
}
} // namespace ninfer::ops::detail
