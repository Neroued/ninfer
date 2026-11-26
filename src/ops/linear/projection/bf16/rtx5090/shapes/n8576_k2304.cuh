#pragma once

#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/kernels/bf16/bf16_schedule.cuh"

namespace ninfer::ops::detail::bf16_n8576_k2304 {

template <class Schedule>
using StaticK = Bf16ScheduleInstance<Schedule, 2304>;

using Gemv = Bf16KTailSchedule<
    StaticK<Bf16A16GemvSchedule<8, 4, 2, 8, 4, Bf16ActivationAccess::Direct,
                                Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>>;
using SlicedK    = StaticK<Bf16A16SlicedKMmaSchedule<64, 8, 2, 64, 2>>;
using Tma32      = StaticK<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 4, 1>>;
using Tma48WideK = StaticK<Bf16A16TmaMmaSchedule<128, 48, 128, 32, 16, 2, 1>>;
using Tma64Deep  = StaticK<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 3, 1>>;
using Tma80      = StaticK<Bf16A16TmaMmaSchedule<128, 80, 64, 32, 16, 2, 2>>;
using Tma48Grouped =
    StaticK<Bf16A16TmaMmaSchedule<128, 48, 64, 32, 16, 2, 2, Bf16MmaRaster::Grouped, 4>>;
using Tma64Grouped =
    StaticK<Bf16A16TmaMmaSchedule<128, 64, 64, 64, 32, 2, 1, Bf16MmaRaster::Grouped, 4>>;
using Tma96Grouped =
    StaticK<Bf16A16TmaMmaSchedule<64, 96, 64, 32, 24, 2, 2, Bf16MmaRaster::Grouped, 4>>;

template <class Routes>
auto select(int tokens, const Routes& routes) {
    if (tokens == 1) return routes.template gemv<Gemv>();
    if (tokens <= 16) return routes.template sliced<SlicedK>();
    if (tokens <= 64) return routes.template tma<Tma32>();
    if (tokens <= 96) return routes.template tma<Tma48WideK, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 128) return routes.template tma<Tma64Deep, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 160) return routes.template tma<Tma80, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 240) return routes.template tma<Tma48Grouped, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 320) return routes.template tma<Tma64Grouped, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 336) return routes.template tma<Tma48Grouped, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 400) return routes.template tma<Tma80, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 512) return routes.template tma<Tma96Grouped, Bf16TmaVectorEpilogue>();
    return routes.template tma<Tma64Grouped, Bf16MmaWarpStoreEpilogue>();
}

} // namespace ninfer::ops::detail::bf16_n8576_k2304
