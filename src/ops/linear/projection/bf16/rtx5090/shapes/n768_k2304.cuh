#pragma once

#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/kernels/bf16/bf16_schedule.cuh"

namespace ninfer::ops::detail::bf16_n768_k2304 {

template <class Schedule>
using StaticK = Bf16ScheduleInstance<Schedule, 2304>;

using Gemv =
    StaticK<Bf16A16GemvSchedule<1, 1, 1, 8, 4, Bf16ActivationAccess::Direct,
                                Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>;
using Sliced8 = Bf16KTailSchedule<StaticK<Bf16A16SlicedKMmaSchedule<16, 8, 4, 128, 3>>>;
using Sliced16 =
    Bf16KTailSchedule<StaticK<Bf16A16SlicedKMmaSchedule<16, 16, 4, 128, 2, Cache::cg, Cache::cg, 1,
                                                        0, Bf16MmaRaster::TokenFast>>>;
using Mma16 =
    StaticK<Bf16A16MmaSchedule<32, 16, 256, 16, 8, 3, 1, Cache::cg, Cache::cg,
                               Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Tma32x32 = StaticK<Bf16A16TmaMmaSchedule<32, 32, 128, 16, 16, 3>>;
using Tma64x32 = StaticK<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 4>>;
using Tma48x64 =
    StaticK<Bf16A16TmaMmaSchedule<48, 64, 64, 16, 16, 4, 1, Bf16MmaRaster::TokenFast, 1, 2>>;
using Tma64x64 =
    StaticK<Bf16A16TmaMmaSchedule<64, 64, 64, 16, 16, 4, 1, Bf16MmaRaster::Grouped, 8>>;
using Tma64x80 = StaticK<Bf16A16TmaMmaSchedule<64, 80, 64, 16, 40, 3>>;
using Tma64x96 =
    StaticK<Bf16A16TmaMmaSchedule<64, 96, 64, 32, 24, 3, 1, Bf16MmaRaster::Grouped, 4>>;
using Tma64x128 =
    StaticK<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 3, 1, Bf16MmaRaster::TokenFast, 1, 3>>;
using Tma128x80 =
    StaticK<Bf16A16TmaMmaSchedule<128, 80, 64, 32, 16, 3, 1, Bf16MmaRaster::TokenFast, 1, 2>>;
using Tma96x128 = StaticK<Bf16A16TmaMmaSchedule<96, 128, 64, 48, 32, 2>>;
using Tma128x112 =
    StaticK<Bf16A16TmaMmaSchedule<128, 112, 64, 32, 56, 3, 1, Bf16MmaRaster::Grouped, 4>>;
using Tma128x160 = StaticK<Bf16A16TmaMmaSchedule<128, 160, 64, 64, 40, 2>>;

template <class Routes>
auto select(int tokens, const Routes& routes) {
    if (tokens == 1) return routes.template gemv<Gemv>();
    if (tokens <= 24) return routes.template sliced<Sliced8>();
    if (tokens <= 48) return routes.template sliced<Sliced16>();
    if (tokens <= 80) return routes.template mma<Mma16>();
    if (tokens <= 224) return routes.template tma<Tma32x32>();
    if (tokens <= 384) return routes.template tma<Tma64x32>();
    if (tokens <= 640) return routes.template tma<Tma48x64, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 896) return routes.template tma<Tma64x64, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 1120) return routes.template tma<Tma64x80, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 1344) return routes.template tma<Tma64x96, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 1792) return routes.template tma<Tma64x128, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 2240) return routes.template tma<Tma128x80, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 2688) return routes.template tma<Tma96x128, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 3136) return routes.template tma<Tma128x112, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 3584) return routes.template tma<Tma64x128, Bf16MmaWarpStoreEpilogue>();
    return routes.template tma<Tma128x160, Bf16MmaWarpStoreEpilogue>();
}

} // namespace ninfer::ops::detail::bf16_n768_k2304
