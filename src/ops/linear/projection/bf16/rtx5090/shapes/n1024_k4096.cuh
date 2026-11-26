#pragma once

#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/kernels/bf16/bf16_schedule.cuh"

namespace ninfer::ops::detail::bf16_n1024_k4096 {

template <class Schedule>
using StaticK = Bf16ScheduleInstance<Schedule, 4096>;

using Gemv =
    StaticK<Bf16A16GemvSchedule<8, 4, 1, 8, 4, Bf16ActivationAccess::Direct,
                                Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>;
using Sliced16x16 = StaticK<Bf16A16SlicedKMmaSchedule<16, 16, 4, 64, 2, Cache::cg, Cache::cg, 1, 1,
                                                      Bf16MmaRaster::TokenFast>>;
using Sliced32x16 = StaticK<Bf16A16SlicedKMmaSchedule<32, 16, 4, 128, 2, Cache::cg, Cache::cg, 1, 1,
                                                      Bf16MmaRaster::TokenFast>>;
using Tma32x32 =
    StaticK<Bf16A16TmaMmaSchedule<32, 32, 128, 16, 16, 3, 2, Bf16MmaRaster::TokenFast, 1, 1>>;
using Tma64x32 =
    StaticK<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 4, 2, Bf16MmaRaster::Grouped, 4, 1>>;
using Tma64x48 =
    StaticK<Bf16A16TmaMmaSchedule<64, 48, 64, 16, 16, 4, 1, Bf16MmaRaster::TokenFast, 1, 1>>;
using Tma64x64 =
    StaticK<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2, Bf16MmaRaster::Grouped, 4, 1>>;
using Tma64x80 =
    StaticK<Bf16A16TmaMmaSchedule<64, 80, 64, 16, 40, 3, 1, Bf16MmaRaster::Grouped, 8, 3>>;
using Tma128x48 =
    StaticK<Bf16A16TmaMmaSchedule<128, 48, 64, 32, 16, 3, 1, Bf16MmaRaster::TokenFast, 1, 2>>;
using Tma128x56 =
    StaticK<Bf16A16TmaMmaSchedule<128, 56, 64, 32, 56, 3, 1, Bf16MmaRaster::Grouped, 8, 3>>;
using Tma128x64 =
    StaticK<Bf16A16TmaMmaSchedule<128, 64, 64, 64, 16, 3, 1, Bf16MmaRaster::TokenFast, 1, 1>>;
using Tma64x160 =
    StaticK<Bf16A16TmaMmaSchedule<64, 160, 64, 32, 40, 3, 1, Bf16MmaRaster::Grouped, 8, 2>>;
using Tma128x96 =
    StaticK<Bf16A16TmaMmaSchedule<128, 96, 64, 64, 24, 2, 1, Bf16MmaRaster::RowFast, 1, 3, 0>>;
using Tma128x112 =
    StaticK<Bf16A16TmaMmaSchedule<128, 112, 64, 32, 56, 3, 1, Bf16MmaRaster::Grouped, 4, 1>>;
using Tma128x128 =
    StaticK<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 8, 3>>;
using Tma128x160 =
    StaticK<Bf16A16TmaMmaSchedule<128, 160, 64, 64, 40, 2, 1, Bf16MmaRaster::Grouped, 4, 3, 0>>;

template <class Routes>
auto select(int tokens, const Routes& routes) {
    if (tokens == 1) return routes.template gemv<Gemv>();
    if (tokens <= 34) return routes.template sliced<Sliced16x16>();
    if (tokens <= 72) return routes.template sliced<Sliced32x16>();
    if (tokens <= 128) return routes.template tma<Tma32x32>();
    if (tokens <= 256) return routes.template tma<Tma64x32>();
    if (tokens <= 480) return routes.template tma<Tma64x48>();
    if (tokens <= 640) return routes.template tma<Tma64x64, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 800) return routes.template tma<Tma64x80, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 1008) return routes.template tma<Tma128x48, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 1176) return routes.template tma<Tma128x56, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 1344) return routes.template tma<Tma128x64, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 1600) return routes.template tma<Tma64x160, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 2016) return routes.template tma<Tma128x96, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 2352) return routes.template tma<Tma128x112, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 2688) return routes.template tma<Tma128x128, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 3360) return routes.template tma<Tma128x160, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 4032) return routes.template tma<Tma128x96, Bf16MmaWarpStoreEpilogue>();
    return routes.template tma<Tma64x80, Bf16MmaWarpStoreEpilogue>();
}

} // namespace ninfer::ops::detail::bf16_n1024_k4096
