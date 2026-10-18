#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Schedule>
using Static = Bf16ScheduleInstance<Schedule, 4096>;
using Gemv =
    Static<Bf16A16GemvSchedule<4, 2, 1, 8, 4, Bf16ActivationAccess::Direct,
                               Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>;
using Short = Static<Bf16A16SlicedKMmaSchedule<16, 8, 8, 64, 2>>;
using Small = Static<Bf16A16SlicedKMmaSchedule<16, 16, 4, 64, 2>>;
using Narrow =
    Static<Bf16A16MmaSchedule<32, 32, 128, 16, 8, 3, 1, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Medium =
    Static<Bf16A16MmaSchedule<64, 32, 64, 16, 16, 2, 2, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Large     = Static<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 1>>;
using Bulk      = Static<Bf16A16TmaMmaSchedule<128, 128, 64, 32, 32, 2, 1>>;
using Streaming = Static<Bf16A16TmaMmaSchedule<128, 128, 64, 32, 32, 2, 1, Bf16MmaRaster::RowFast>>;
} // namespace

Bf16Launch select_bf16_n2048_k4096(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Gemv>;
    if (tokens <= 8) return launch_bf16_sliced_k_mma<Short>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<Small>;
    if (tokens <= 32) return launch_bf16_mma<Narrow>;
    if (tokens <= 128) return launch_bf16_mma<Medium>;
    if (tokens <= 512) return launch_bf16_tma_mma<Large>;
    if (tokens <= 2048) return launch_bf16_tma_mma<Bulk>;
    return launch_bf16_tma_mma<Streaming>;
}
} // namespace ninfer::ops::detail
