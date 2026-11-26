#pragma once

#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/kernels/bf16/bf16_schedule.cuh"

namespace ninfer::ops::detail::bf16_n1024_k2304 {

template <class Schedule>
using StaticK = Bf16ScheduleInstance<Schedule, 2304>;

using Gemv =
    StaticK<Bf16A16GemvSchedule<8, 1, 2, 8, 4, Bf16ActivationAccess::Direct,
                                Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 2, 2>>;
using Sliced8 = Bf16KTailSchedule<
    StaticK<Bf16A16SlicedKMmaSchedule<16, 8, 4, 128, 2, Cache::cg, Cache::cg, 1, 1>>>;
using Mma16 =
    StaticK<Bf16A16MmaSchedule<32, 16, 256, 16, 8, 3, 1, Cache::cg, Cache::cg,
                               Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Mma32 =
    StaticK<Bf16A16MmaSchedule<32, 32, 256, 16, 8, 3, 1, Cache::cg, Cache::cg,
                               Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Tma32 = StaticK<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 5, 1, Bf16MmaRaster::Grouped, 4>>;
using Tma48 = StaticK<Bf16A16TmaMmaSchedule<64, 48, 64, 16, 16, 3, 1, Bf16MmaRaster::Grouped, 4>>;
using Tma64 = StaticK<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 1, Bf16MmaRaster::Grouped, 4>>;
using Tma48Wide =
    StaticK<Bf16A16TmaMmaSchedule<128, 48, 64, 32, 16, 4, 1, Bf16MmaRaster::Grouped, 4, 2>>;
using Tma80 =
    StaticK<Bf16A16TmaMmaSchedule<128, 80, 64, 32, 16, 3, 1, Bf16MmaRaster::TokenFast, 1, 2>>;
using Tma112 =
    StaticK<Bf16A16TmaMmaSchedule<128, 112, 64, 32, 56, 3, 1, Bf16MmaRaster::TokenFast, 1, 2>>;
using Tma128 = StaticK<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 3>>;

template <class Routes>
auto select(int tokens, const Routes& routes) {
    if (tokens == 1) return routes.template gemv<Gemv>();
    if (tokens <= 24) return routes.template sliced<Sliced8>();
    if (tokens <= 80) return routes.template mma<Mma16>();
    if (tokens <= 144) return routes.template mma<Mma32>();
    if (tokens <= 320) return routes.template tma<Tma32, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 480) return routes.template tma<Tma48, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 640) return routes.template tma<Tma64, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 912) return routes.template tma<Tma48Wide, Bf16TmaVectorEpilogue>();
    if (tokens <= 1280) return routes.template tma<Tma128>();
    if (tokens <= 1680) return routes.template tma<Tma80, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 1920) return routes.template tma<Tma64, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 2352) return routes.template tma<Tma112, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 2560) return routes.template tma<Tma128>();
    if (tokens <= 3360) return routes.template tma<Tma80, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 3840) return routes.template tma<Tma128>();
    return routes.template tma<Tma112, Bf16MmaWarpStoreEpilogue>();
}

} // namespace ninfer::ops::detail::bf16_n1024_k2304
