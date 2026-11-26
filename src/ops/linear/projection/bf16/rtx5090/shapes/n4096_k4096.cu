#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {

template <class Schedule>
using StaticK = Bf16ScheduleInstance<Schedule, 4096>;

using Gemv =
    StaticK<Bf16A16GemvSchedule<16, 4, 2, 4, 4, Bf16ActivationAccess::Direct,
                                Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>;
using Mma32x16 =
    StaticK<Bf16A16MmaSchedule<32, 16, 128, 16, 8, 3, 2, Cache::cg, Cache::cg,
                               Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Tma64x32 =
    StaticK<Bf16A16TmaMmaSchedule<64, 32, 128, 32, 16, 3, 1, Bf16MmaRaster::RowFast, 1, 1, 1>>;
using Tma64x48      = StaticK<Bf16A16TmaMmaSchedule<64, 48, 128, 16, 16, 3, 1>>;
using Tma64x64WideK = StaticK<Bf16A16TmaMmaSchedule<64, 64, 128, 16, 32, 3, 1>>;
using Tma64x80Deep =
    StaticK<Bf16A16TmaMmaSchedule<64, 80, 64, 16, 40, 3, 1, Bf16MmaRaster::Grouped, 8, 3>>;
using Tma128x48 =
    StaticK<Bf16A16TmaMmaSchedule<128, 48, 64, 32, 16, 3, 1, Bf16MmaRaster::RowFast, 1, 3>>;
using Tma128x56 =
    StaticK<Bf16A16TmaMmaSchedule<128, 56, 64, 32, 56, 2, 2, Bf16MmaRaster::RowFast, 1, 3>>;
using Tma64x64 =
    StaticK<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2, Bf16MmaRaster::RowFast, 1, 3>>;
using Tma128x80 =
    StaticK<Bf16A16TmaMmaSchedule<128, 80, 64, 32, 16, 3, 1, Bf16MmaRaster::Grouped, 4, 2>>;
using Tma128x112Deep =
    StaticK<Bf16A16TmaMmaSchedule<128, 112, 64, 32, 56, 3, 1, Bf16MmaRaster::Grouped, 4, 2>>;
using Tma128x128 =
    StaticK<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 2, 1, Bf16MmaRaster::Grouped, 4, 2>>;
using Tma128x160 = StaticK<Bf16A16TmaMmaSchedule<128, 160, 64, 64, 40, 2, 1>>;
using Tma64x80 =
    StaticK<Bf16A16TmaMmaSchedule<64, 80, 64, 16, 40, 2, 2, Bf16MmaRaster::Grouped, 4, 2>>;
using Tma128x112 =
    StaticK<Bf16A16TmaMmaSchedule<128, 112, 64, 32, 56, 2, 1, Bf16MmaRaster::Grouped, 4, 2>>;

} // namespace

Bf16Launch select_bf16_n4096_k4096(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Gemv>;
    if (tokens <= 32) return launch_bf16_mma<Mma32x16>;
    if (tokens <= 64) return launch_bf16_tma_mma<Tma64x32, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 96) return launch_bf16_tma_mma<Tma64x48>;
    if (tokens <= 128) return launch_bf16_tma_mma<Tma64x64WideK, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 160) return launch_bf16_tma_mma<Tma64x80Deep, Bf16TmaVectorEpilogue>;
    if (tokens <= 240) return launch_bf16_tma_mma<Tma128x48, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 280) return launch_bf16_tma_mma<Tma128x56, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 320) return launch_bf16_tma_mma<Tma64x64, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 400) return launch_bf16_tma_mma<Tma128x80, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 560) return launch_bf16_tma_mma<Tma128x112Deep, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 640) return launch_bf16_tma_mma<Tma128x128, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 800) return launch_bf16_tma_mma<Tma128x160, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 1040) return launch_bf16_tma_mma<Tma64x80, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 1344) return launch_bf16_tma_mma<Tma128x56, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 1600) return launch_bf16_tma_mma<Tma128x160, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 3920) return launch_bf16_tma_mma<Tma128x56, Bf16MmaWarpStoreEpilogue>;
    return launch_bf16_tma_mma<Tma128x112, Bf16MmaWarpStoreEpilogue>;
}

} // namespace ninfer::ops::detail
