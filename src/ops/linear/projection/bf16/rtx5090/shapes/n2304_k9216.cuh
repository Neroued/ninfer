#pragma once

#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/kernels/bf16/bf16_schedule.cuh"

namespace ninfer::ops::detail::bf16_n2304_k9216 {

template <class Schedule>
using StaticK = Bf16ScheduleInstance<Schedule, 9216>;

using Gemv =
    StaticK<Bf16A16GemvSchedule<8, 2, 1, 16, 4, Bf16ActivationAccess::Direct,
                                Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 2, 2>>;
using Sliced8   = StaticK<Bf16A16SlicedKMmaSchedule<32, 8, 8, 64, 2>>;
using Sliced16  = StaticK<Bf16A16SlicedKMmaSchedule<16, 16, 4, 64, 2, Cache::cg, Cache::cg, 1, 1,
                                                    Bf16MmaRaster::TokenFast>>;
using Sliced32  = StaticK<Bf16A16SlicedKMmaSchedule<32, 32, 4, 64, 2, Cache::cg, Cache::cg, 1, 1,
                                                    Bf16MmaRaster::TokenFast>>;
using Sliced24  = StaticK<Bf16A16SlicedKMmaSchedule<48, 24, 8, 32, 2, Cache::cg, Cache::cg, 1, 1,
                                                    Bf16MmaRaster::TokenFast>>;
using Sliced40  = StaticK<Bf16A16SlicedKMmaSchedule<32, 40, 8, 32, 2, Cache::cg, Cache::cg, 1, 1,
                                                    Bf16MmaRaster::TokenFast>>;
using TmaR32T48 = StaticK<Bf16A16TmaMmaSchedule<32, 48, 128, 16, 24, 3, 1>>;
using TmaR96T32 =
    StaticK<Bf16A16TmaMmaSchedule<96, 32, 64, 16, 16, 4, 1, Bf16MmaRaster::TokenFast, 1, 4>>;
using TmaR64T64 =
    StaticK<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 4, 1, Bf16MmaRaster::TokenFast, 1, 4>>;
using TmaR96T64 = StaticK<Bf16A16TmaMmaSchedule<96, 64, 64, 48, 16, 3, 1>>;
using TmaR128T64 =
    StaticK<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 3, 1, Bf16MmaRaster::RowFast>>;
using TmaR256T64 =
    StaticK<Bf16A16TmaMmaSchedule<256, 64, 64, 64, 32, 2, 1, Bf16MmaRaster::TokenFast, 1, 4>>;

template <class Routes>
auto select(int tokens, const Routes& routes) {
    if (tokens == 1) return routes.template gemv<Gemv>();
    if (tokens <= 16) return routes.template sliced<Sliced8>();
    if (tokens <= 32) return routes.template sliced<Sliced16>();
    if (tokens <= 64) return routes.template sliced<Sliced32>();
    if (tokens <= 72) return routes.template sliced<Sliced24>();
    if (tokens <= 80) return routes.template sliced<Sliced40>();
    if (tokens <= 96) return routes.template tma<TmaR32T48, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 192) return routes.template tma<TmaR96T32, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 256) return routes.template tma<TmaR64T64, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 384) return routes.template tma<TmaR96T64>();
    if (tokens <= 576) return routes.template tma<TmaR128T64>();
    if (tokens <= 768) return routes.template tma<TmaR96T64>();
    if (tokens <= 1152) return routes.template tma<TmaR256T64, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 1280) return routes.template tma<TmaR96T64>();
    if (tokens <= 1728) return routes.template tma<TmaR128T64>();
    if (tokens <= 3072) return routes.template tma<TmaR96T64>();
    return routes.template tma<TmaR128T64>();
}

} // namespace ninfer::ops::detail::bf16_n2304_k9216
