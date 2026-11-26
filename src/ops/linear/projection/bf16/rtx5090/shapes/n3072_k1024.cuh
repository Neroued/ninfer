#pragma once

#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/kernels/bf16/bf16_schedule.cuh"

namespace ninfer::ops::detail::bf16_n3072_k1024 {

template <class Schedule>
using StaticK = Bf16ScheduleInstance<Schedule, 1024>;

using Gemv =
    StaticK<Bf16A16GemvSchedule<8, 4, 2, 8, 4, Bf16ActivationAccess::Direct,
                                Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>;
using Sliced16x8 = StaticK<
    Bf16A16SlicedKMmaSchedule<16, 8, 2, 64, 2, Cache::cg, Cache::cg, 1, 1, Bf16MmaRaster::RowFast>>;
using Tma64x16 =
    StaticK<Bf16A16TmaMmaSchedule<64, 16, 64, 16, 8, 4, 2, Bf16MmaRaster::TokenFast, 1, 1>>;
using Tma64x32 =
    StaticK<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 4, 2, Bf16MmaRaster::TokenFast, 1, 1>>;
using Tma64x48 =
    StaticK<Bf16A16TmaMmaSchedule<64, 48, 64, 16, 16, 3, 1, Bf16MmaRaster::TokenFast, 1, 1>>;
using Tma64x64 =
    StaticK<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3, 1, Bf16MmaRaster::Grouped, 4, 3, 2>>;
using Tma128x48 =
    StaticK<Bf16A16TmaMmaSchedule<128, 48, 64, 32, 16, 3, 1, Bf16MmaRaster::TokenFast, 1, 2>>;
using Tma128x64Phased =
    StaticK<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 3, 1, Bf16MmaRaster::RowFast, 1, 3, 2>>;
using Tma128x80 =
    StaticK<Bf16A16TmaMmaSchedule<128, 80, 64, 32, 16, 3, 1, Bf16MmaRaster::Grouped, 8, 2>>;
using Tma128x96 =
    StaticK<Bf16A16TmaMmaSchedule<128, 96, 64, 64, 24, 2, 1, Bf16MmaRaster::Grouped, 4, 3, 0>>;
using Tma128x56TokenFast =
    StaticK<Bf16A16TmaMmaSchedule<128, 56, 64, 32, 56, 2, 2, Bf16MmaRaster::TokenFast, 1, 2>>;
using Tma128x64 =
    StaticK<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 1, Bf16MmaRaster::RowFast, 1, 3, 0>>;
using Tma128x160 =
    StaticK<Bf16A16TmaMmaSchedule<128, 160, 64, 64, 40, 2, 1, Bf16MmaRaster::Grouped, 4, 3, 0>>;
using Tma128x192 =
    StaticK<Bf16A16TmaMmaSchedule<128, 192, 64, 64, 48, 2, 1, Bf16MmaRaster::Grouped, 4, 2>>;
using Tma128x224 =
    StaticK<Bf16A16TmaMmaSchedule<128, 224, 64, 64, 56, 2, 1, Bf16MmaRaster::Grouped, 4, 2>>;
using Tma128x160RowFast =
    StaticK<Bf16A16TmaMmaSchedule<128, 160, 64, 64, 40, 2, 1, Bf16MmaRaster::RowFast, 1, 3, 0>>;
using Tma128x56 =
    StaticK<Bf16A16TmaMmaSchedule<128, 56, 64, 32, 56, 2, 2, Bf16MmaRaster::Grouped, 2, 1>>;

template <class Routes>
auto select(int tokens, const Routes& routes) {
    if (tokens == 1) return routes.template gemv<Gemv>();
    if (tokens <= 16) return routes.template sliced<Sliced16x8>();
    if (tokens <= 48) return routes.template tma<Tma64x16>();
    if (tokens <= 80) return routes.template tma<Tma64x32>();
    if (tokens <= 144) return routes.template tma<Tma64x48>();
    if (tokens <= 192) return routes.template tma<Tma64x64, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 336) return routes.template tma<Tma128x48, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 448) return routes.template tma<Tma128x64Phased, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 560) return routes.template tma<Tma128x80, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 672) return routes.template tma<Tma128x96, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 784) return routes.template tma<Tma128x56TokenFast, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 896) return routes.template tma<Tma128x64, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 1120) return routes.template tma<Tma128x160, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 1344) return routes.template tma<Tma128x192, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 1568) return routes.template tma<Tma128x224, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 1792) return routes.template tma<Tma128x64, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 1920) return routes.template tma<Tma128x56TokenFast, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 2016) return routes.template tma<Tma128x96, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 2240) return routes.template tma<Tma128x160RowFast, Bf16MmaWarpStoreEpilogue>();
    return routes.template tma<Tma128x56, Bf16MmaWarpStoreEpilogue>();
}

} // namespace ninfer::ops::detail::bf16_n3072_k1024
