#pragma once

#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/kernels/bf16/bf16_schedule.cuh"

namespace ninfer::ops::detail::bf16_n2048_k2304 {

template <class Schedule>
using StaticK = Bf16ScheduleInstance<Schedule, 2304>;

using Sliced8  = StaticK<Bf16A16SlicedKMmaSchedule<32, 8, 4, 64, 2, Cache::cg, Cache::cg, 1, 1>>;
using Sliced16 = StaticK<Bf16A16SlicedKMmaSchedule<32, 16, 4, 64, 2, Cache::cg, Cache::cg, 1, 1,
                                                   Bf16MmaRaster::TokenFast>>;
using Sliced40 = StaticK<Bf16A16SlicedKMmaSchedule<32, 40, 4, 64, 2, Cache::cg, Cache::cg, 1, 1,
                                                   Bf16MmaRaster::TokenFast>>;
using Tma32 =
    StaticK<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 4, 1, Bf16MmaRaster::TokenFast, 1, 4>>;
using Tma64 =
    StaticK<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 1, Bf16MmaRaster::Grouped, 8, 2>>;
using Tma48 =
    StaticK<Bf16A16TmaMmaSchedule<128, 48, 64, 32, 16, 4, 1, Bf16MmaRaster::TokenFast, 1, 2>>;
using Tma80Wide =
    StaticK<Bf16A16TmaMmaSchedule<128, 80, 64, 32, 16, 3, 1, Bf16MmaRaster::RowFast, 1, 2>>;
using Tma104 =
    StaticK<Bf16A16TmaMmaSchedule<128, 104, 64, 32, 104, 2, 1, Bf16MmaRaster::TokenFast, 1, 4>>;
using Tma80 =
    StaticK<Bf16A16TmaMmaSchedule<64, 80, 64, 32, 40, 2, 1, Bf16MmaRaster::Grouped, 4, 2>>;
using Tma112 =
    StaticK<Bf16A16TmaMmaSchedule<64, 112, 64, 32, 56, 2, 1, Bf16MmaRaster::Grouped, 4, 2>>;

template <class Routes>
auto select(int tokens, const Routes& routes) {
    if (tokens <= 16) return routes.template sliced<Sliced8>();
    if (tokens <= 48) return routes.template sliced<Sliced16>();
    if (tokens <= 72) return routes.template sliced<Sliced40>();
    if (tokens <= 160) return routes.template tma<Tma32>();
    if (tokens <= 320) return routes.template tma<Tma64, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 480) return routes.template tma<Tma48, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 640) return routes.template tma<Tma64, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 800) return routes.template tma<Tma80Wide>();
    if (tokens <= 1040) return routes.template tma<Tma104, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 2112) return routes.template tma<Tma80, Bf16MmaWarpStoreEpilogue>();
    return routes.template tma<Tma112, Bf16MmaWarpStoreEpilogue>();
}

} // namespace ninfer::ops::detail::bf16_n2048_k2304
