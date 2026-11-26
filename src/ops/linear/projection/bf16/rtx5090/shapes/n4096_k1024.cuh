#pragma once

#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/kernels/bf16/bf16_schedule.cuh"

namespace ninfer::ops::detail::bf16_n4096_k1024 {

template <class Schedule>
using StaticK = Bf16ScheduleInstance<Schedule, 1024>;

using Gemv =
    StaticK<Bf16A16GemvSchedule<8, 4, 2, 8, 4, Bf16ActivationAccess::Direct,
                                Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 2, 2>>;
using Mma32x24 =
    StaticK<Bf16A16MmaSchedule<32, 24, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                               Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Tma64x32 =
    StaticK<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 4, 2, Bf16MmaRaster::TokenFast, 1, 1>>;
using Tma64x48 =
    StaticK<Bf16A16TmaMmaSchedule<64, 48, 64, 16, 16, 4, 1, Bf16MmaRaster::TokenFast, 1, 1>>;
using Tma128x32 =
    StaticK<Bf16A16TmaMmaSchedule<128, 32, 64, 32, 16, 3, 1, Bf16MmaRaster::TokenFast, 1, 1>>;
using Tma128x48 =
    StaticK<Bf16A16TmaMmaSchedule<128, 48, 64, 32, 16, 3, 1, Bf16MmaRaster::TokenFast, 1, 2>>;
using Tma128x64Phased =
    StaticK<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 3, 1, Bf16MmaRaster::Grouped, 4, 3, 2>>;
using Tma128x64 =
    StaticK<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 1, Bf16MmaRaster::RowFast, 1, 3>>;
using Tma128x80 =
    StaticK<Bf16A16TmaMmaSchedule<128, 80, 64, 32, 16, 3, 1, Bf16MmaRaster::Grouped, 8, 2>>;
using Tma128x96 =
    StaticK<Bf16A16TmaMmaSchedule<128, 96, 64, 64, 24, 2, 1, Bf16MmaRaster::RowFast, 1, 3, 0>>;
using Tma128x112 =
    StaticK<Bf16A16TmaMmaSchedule<128, 112, 64, 32, 56, 3, 1, Bf16MmaRaster::Grouped, 4, 1>>;
using Tma64x128 =
    StaticK<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2, 1, Bf16MmaRaster::RowFast, 1, 3, 0>>;
using Tma128x160 =
    StaticK<Bf16A16TmaMmaSchedule<128, 160, 64, 64, 40, 2, 1, Bf16MmaRaster::RowFast, 1, 3, 0>>;
using Tma64x80 =
    StaticK<Bf16A16TmaMmaSchedule<64, 80, 64, 16, 40, 2, 2, Bf16MmaRaster::Grouped, 4, 2>>;
using Tma128x56 =
    StaticK<Bf16A16TmaMmaSchedule<128, 56, 64, 32, 56, 2, 2, Bf16MmaRaster::TokenFast, 1, 2>>;

template <class Routes>
auto select(int tokens, const Routes& routes) {
    if (tokens == 1) return routes.template gemv<Gemv>();
    if (tokens <= 47) return routes.template mma<Mma32x24>();
    if (tokens <= 63) return routes.template tma<Tma64x32>();
    if (tokens <= 96) return routes.template tma<Tma64x48>();
    if (tokens <= 160) return routes.template tma<Tma128x32>();
    if (tokens <= 240) return routes.template tma<Tma128x48, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 320) return routes.template tma<Tma128x64Phased, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 400) return routes.template tma<Tma128x80, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 480) return routes.template tma<Tma128x96, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 560) return routes.template tma<Tma128x112, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 640) return routes.template tma<Tma64x128, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 800) return routes.template tma<Tma128x160, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 960) return routes.template tma<Tma128x64, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 1040) return routes.template tma<Tma64x80, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 1456) return routes.template tma<Tma128x56, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 1600) return routes.template tma<Tma128x160, Bf16MmaWarpStoreEpilogue>();
    return routes.template tma<Tma128x56, Bf16MmaWarpStoreEpilogue>();
}

} // namespace ninfer::ops::detail::bf16_n4096_k1024
