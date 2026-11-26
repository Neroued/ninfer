#pragma once

#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/kernels/bf16/bf16_schedule.cuh"

namespace ninfer::ops::detail::bf16_n2304_k3072 {

template <class Schedule>
using StaticK = Bf16ScheduleInstance<Schedule, 3072>;

using Gemv = StaticK<
    Bf16A16GemvSchedule<8, 2, 2, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Streaming,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>;
using Sliced8WideK = StaticK<Bf16A16SlicedKMmaSchedule<16, 8, 12, 64, 2>>;
using Sliced8      = StaticK<Bf16A16SlicedKMmaSchedule<16, 8, 8, 64, 2>>;
using Sliced16     = StaticK<Bf16A16SlicedKMmaSchedule<48, 16, 4, 64, 2, Cache::cg, Cache::cg, 1, 1,
                                                       Bf16MmaRaster::TokenFast>>;
using Sliced32     = StaticK<Bf16A16SlicedKMmaSchedule<32, 32, 4, 64, 2, Cache::cg, Cache::cg, 1, 1,
                                                       Bf16MmaRaster::TokenFast>>;
using Sliced24     = StaticK<Bf16A16SlicedKMmaSchedule<64, 24, 8, 32, 2, Cache::cg, Cache::cg, 1, 0,
                                                       Bf16MmaRaster::TokenFast>>;
using TmaR64T32 =
    StaticK<Bf16A16TmaMmaSchedule<64, 32, 128, 16, 16, 2, 1, Bf16MmaRaster::TokenFast, 1, 1, 1>>;
using TmaR96T32 =
    StaticK<Bf16A16TmaMmaSchedule<96, 32, 64, 16, 16, 3, 1, Bf16MmaRaster::TokenFast, 1, 1, 1>>;
using TmaR64T64 =
    StaticK<Bf16A16TmaMmaSchedule<64, 64, 128, 32, 32, 2, 1, Bf16MmaRaster::TokenFast, 1, 1, 1>>;
using TmaR96T64 = StaticK<Bf16A16TmaMmaSchedule<96, 64, 64, 48, 16, 3, 1>>;
using TmaR128T64 =
    StaticK<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 1, Bf16MmaRaster::Grouped, 4, 2>>;
using TmaR256T64 =
    StaticK<Bf16A16TmaMmaSchedule<256, 64, 64, 64, 32, 2, 1, Bf16MmaRaster::TokenFast, 1, 4>>;

template <class Routes>
auto select(int tokens, const Routes& routes) {
    if (tokens == 1) return routes.template gemv<Gemv>();
    if (tokens <= 8) return routes.template sliced<Sliced8WideK>();
    if (tokens <= 16) return routes.template sliced<Sliced8>();
    if (tokens <= 24) return routes.template sliced<Sliced16>();
    if (tokens <= 64) return routes.template sliced<Sliced32>();
    if (tokens <= 72) return routes.template sliced<Sliced24>();
    if (tokens <= 128) return routes.template tma<TmaR64T32, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 192) return routes.template tma<TmaR96T32, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 256) return routes.template tma<TmaR64T64, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 384) return routes.template tma<TmaR96T64>();
    if (tokens <= 576) return routes.template tma<TmaR128T64, Bf16TmaVectorEpilogue>();
    if (tokens <= 768) return routes.template tma<TmaR96T64>();
    if (tokens <= 1152) return routes.template tma<TmaR256T64, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 1280) return routes.template tma<TmaR96T64>();
    return routes.template tma<TmaR128T64, Bf16TmaVectorEpilogue>();
}

} // namespace ninfer::ops::detail::bf16_n2304_k3072
