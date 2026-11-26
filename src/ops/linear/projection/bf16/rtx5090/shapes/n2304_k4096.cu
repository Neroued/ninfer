#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {

template <class Schedule>
using StaticK = Bf16ScheduleInstance<Schedule, 4096>;

using Gemv =
    StaticK<Bf16A16GemvSchedule<1, 1, 1, 8, 4, Bf16ActivationAccess::Direct,
                                Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>;
using Sliced48x8 = StaticK<Bf16A16SlicedKMmaSchedule<48, 8, 4, 64, 2, Cache::cg, Cache::cg>>;
using Tma64x16 =
    StaticK<Bf16A16TmaMmaSchedule<64, 16, 128, 16, 8, 3, 1, Bf16MmaRaster::Grouped, 4, 2>>;
using Tma32x32 =
    StaticK<Bf16A16TmaMmaSchedule<32, 32, 128, 16, 16, 3, 2, Bf16MmaRaster::RowFast, 1, 1, 1>>;
using Tma64x32 = StaticK<Bf16A16TmaMmaSchedule<64, 32, 128, 16, 16, 3, 1>>;
using Tma64x48 = StaticK<Bf16A16TmaMmaSchedule<64, 48, 64, 16, 16, 4, 1>>;
using Tma64x64 = StaticK<Bf16A16TmaMmaSchedule<64, 64, 64, 16, 32, 3, 2>>;
using Tma64x80 =
    StaticK<Bf16A16TmaMmaSchedule<64, 80, 64, 16, 40, 3, 1, Bf16MmaRaster::RowFast, 1, 3>>;
using Tma96x64 =
    StaticK<Bf16A16TmaMmaSchedule<96, 64, 64, 48, 16, 3, 1, Bf16MmaRaster::RowFast, 1, 3>>;
using Tma128x64 =
    StaticK<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 3, 1, Bf16MmaRaster::Grouped, 8, 3>>;
using Tma128x128 = StaticK<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>>;
using Tma128x56 =
    StaticK<Bf16A16TmaMmaSchedule<128, 56, 64, 32, 56, 2, 2, Bf16MmaRaster::RowFast, 1, 3>>;
using Tma128x112 =
    StaticK<Bf16A16TmaMmaSchedule<128, 112, 64, 32, 56, 2, 1, Bf16MmaRaster::Grouped, 4, 2>>;

} // namespace

Bf16Launch select_bf16_n2304_k4096(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Gemv>;
    if (tokens <= 24) return launch_bf16_sliced_k_mma<Sliced48x8>;
    if (tokens <= 48) return launch_bf16_tma_mma<Tma64x16, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 64) return launch_bf16_tma_mma<Tma32x32, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 128) return launch_bf16_tma_mma<Tma64x32>;
    if (tokens <= 192) return launch_bf16_tma_mma<Tma64x48>;
    if (tokens <= 256) return launch_bf16_tma_mma<Tma64x64, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 320) return launch_bf16_tma_mma<Tma64x80, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 448) return launch_bf16_tma_mma<Tma96x64, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 576) return launch_bf16_tma_mma<Tma128x64, Bf16TmaVectorEpilogue>;
    if (tokens <= 896) return launch_bf16_tma_mma<Tma96x64, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 1152) return launch_bf16_tma_mma<Tma128x128, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 2632) return launch_bf16_tma_mma<Tma128x56, Bf16MmaWarpStoreEpilogue>;
    return launch_bf16_tma_mma<Tma128x112, Bf16MmaWarpStoreEpilogue>;
}

} // namespace ninfer::ops::detail
