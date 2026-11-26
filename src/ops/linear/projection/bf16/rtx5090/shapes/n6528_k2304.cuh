#pragma once

#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/kernels/bf16/bf16_schedule.cuh"

namespace ninfer::ops::detail::bf16_n6528_k2304 {

template <class Schedule>
using StaticK = Bf16ScheduleInstance<Schedule, 2304>;

using SlicedK   = StaticK<Bf16A16SlicedKMmaSchedule<32, 8, 3, 64, 2>>;
using Mma16     = StaticK<Bf16A16MmaSchedule<64, 16, 128, 16, 16, 2, 2, Cache::cg, Cache::cg,
                                             Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::Grouped,
                                             Bf16MmaSwizzle::Xor64, 4>>;
using Tma32     = StaticK<Bf16A16TmaMmaSchedule<128, 32, 64, 16, 16, 4, 1>>;
using Tma48     = StaticK<Bf16A16TmaMmaSchedule<128, 48, 64, 32, 16, 4, 1>>;
using Tma64Deep = StaticK<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 3, 1>>;
using Tma80     = StaticK<Bf16A16TmaMmaSchedule<128, 80, 64, 32, 16, 2, 2>>;
using Tma96     = StaticK<Bf16A16TmaMmaSchedule<128, 96, 64, 32, 32, 3, 1>>;
using Tma112    = StaticK<Bf16A16TmaMmaSchedule<128, 112, 64, 32, 56, 2, 1>>;
using Tma144 =
    StaticK<Bf16A16TmaMmaSchedule<128, 144, 64, 32, 48, 2, 1, Bf16MmaRaster::Grouped, 4>>;
using Tma64Narrow =
    StaticK<Bf16A16TmaMmaSchedule<64, 64, 64, 16, 32, 3, 1, Bf16MmaRaster::Grouped, 4>>;
using Tma64Wide =
    StaticK<Bf16A16TmaMmaSchedule<128, 64, 64, 64, 32, 2, 1, Bf16MmaRaster::Grouped, 4>>;

template <class Routes>
auto select(int tokens, const Routes& routes) {
    if (tokens <= 16) return routes.template sliced<SlicedK>();
    if (tokens <= 48) return routes.template mma<Mma16>();
    if (tokens <= 96) return routes.template tma<Tma32>();
    if (tokens <= 144) return routes.template tma<Tma48>();
    if (tokens <= 192) return routes.template tma<Tma64Deep, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 240) return routes.template tma<Tma80, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 288) return routes.template tma<Tma96>();
    if (tokens <= 336) return routes.template tma<Tma112, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 432) return routes.template tma<Tma144, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 512) return routes.template tma<Tma64Narrow, Bf16MmaWarpStoreEpilogue>();
    return routes.template tma<Tma64Wide, Bf16TmaVectorEpilogue>();
}

} // namespace ninfer::ops::detail::bf16_n6528_k2304
