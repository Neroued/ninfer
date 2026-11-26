#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {

template <class Schedule>
using StaticK = Bf16KTailSchedule<Bf16ScheduleInstance<Schedule, 1200>>;

using Gemv =
    StaticK<Bf16A16GemvSchedule<4, 4, 1, 8, 4, Bf16ActivationAccess::Direct,
                                Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2>>;
using Sliced32x8 = StaticK<Bf16A16SlicedKMmaSchedule<32, 8, 8, 32, 2, Cache::cg, Cache::cg>>;
using Tma32x32   = StaticK<Bf16A16TmaMmaSchedule<32, 32, 64, 16, 16, 4, 2>>;
using Tma64x32 =
    StaticK<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 4, 2, Bf16MmaRaster::Grouped, 4, 2>>;
using Tma64x48 =
    StaticK<Bf16A16TmaMmaSchedule<64, 48, 64, 16, 16, 3, 1, Bf16MmaRaster::Grouped, 4, 2>>;
using Tma64x64 =
    StaticK<Bf16A16TmaMmaSchedule<64, 64, 64, 16, 32, 3, 2, Bf16MmaRaster::Grouped, 4, 2>>;
using Tma64x80 =
    StaticK<Bf16A16TmaMmaSchedule<64, 80, 64, 16, 40, 3, 1, Bf16MmaRaster::Grouped, 4, 2>>;
using Tma128x64 =
    StaticK<Bf16A16TmaMmaSchedule<128, 64, 64, 64, 16, 3, 1, Bf16MmaRaster::RowFast, 1, 3>>;
using Tma64x160 =
    StaticK<Bf16A16TmaMmaSchedule<64, 160, 64, 32, 40, 3, 1, Bf16MmaRaster::Grouped, 4, 2>>;
using Tma128x112 =
    StaticK<Bf16A16TmaMmaSchedule<128, 112, 64, 32, 56, 3, 1, Bf16MmaRaster::RowFast, 1, 3>>;
using Tma64x80TwoStage =
    StaticK<Bf16A16TmaMmaSchedule<64, 80, 64, 16, 40, 2, 2, Bf16MmaRaster::Grouped, 4, 2>>;

} // namespace

Bf16Launch select_bf16_n1024_k1200(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Gemv>;
    if (tokens <= 24) return launch_bf16_sliced_k_mma<Sliced32x8>;
    if (tokens <= 128) return launch_bf16_tma_mma<Tma32x32>;
    if (tokens <= 320) return launch_bf16_tma_mma<Tma64x32, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 480) return launch_bf16_tma_mma<Tma64x48, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 640) return launch_bf16_tma_mma<Tma64x64, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 800) return launch_bf16_tma_mma<Tma64x80, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 1344) return launch_bf16_tma_mma<Tma128x64, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 1600) return launch_bf16_tma_mma<Tma64x160, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 2352) return launch_bf16_tma_mma<Tma128x112, Bf16MmaWarpStoreEpilogue>;
    return launch_bf16_tma_mma<Tma64x80TwoStage, Bf16MmaWarpStoreEpilogue>;
}

} // namespace ninfer::ops::detail
