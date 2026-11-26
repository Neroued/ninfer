#pragma once

#include "ops/linear/kernels/bf16/bf16_schedule.cuh"

namespace ninfer::ops::detail::bf16_n5120_k2304 {

template <class Schedule>
using StaticK = Bf16ScheduleInstance<Schedule, 2304>;

using Sliced8 = StaticK<Bf16A16SlicedKMmaSchedule<32, 8, 4, 64, 2, Cache::cg, Cache::cg, 1, 1>>;
using Mma16 =
    StaticK<Bf16A16MmaSchedule<64, 16, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                               Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Tma16 =
    StaticK<Bf16A16TmaMmaSchedule<128, 16, 64, 16, 16, 4, 1, Bf16MmaRaster::RowFast, 1, 2>>;
using Tma32 =
    StaticK<Bf16A16TmaMmaSchedule<128, 32, 64, 16, 16, 4, 1, Bf16MmaRaster::RowFast, 1, 2>>;
using Tma48 =
    StaticK<Bf16A16TmaMmaSchedule<128, 48, 64, 32, 16, 4, 1, Bf16MmaRaster::Grouped, 8, 4>>;
using Tma64 =
    StaticK<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 3, 1, Bf16MmaRaster::TokenFast, 1, 4>>;
using Tma96 =
    StaticK<Bf16A16TmaMmaSchedule<128, 96, 64, 32, 32, 3, 1, Bf16MmaRaster::Grouped, 8, 4>>;
using Tma64Grouped =
    StaticK<Bf16A16TmaMmaSchedule<128, 64, 64, 64, 32, 2, 1, Bf16MmaRaster::Grouped, 8, 4>>;

template <class Routes>
auto select(int tokens, const Routes& routes) {
    if (tokens <= 16) return routes.template sliced<Sliced8>();
    if (tokens <= 32) return routes.template mma<Mma16>();
    if (tokens <= 64) return routes.template tma<Tma16>();
    if (tokens <= 128) return routes.template tma<Tma32>();
    if (tokens <= 192) return routes.template tma<Tma48>();
    if (tokens <= 256) return routes.template tma<Tma64>();
    if (tokens <= 384) return routes.template tma<Tma96>();
    return routes.template tma<Tma64Grouped>();
}

} // namespace ninfer::ops::detail::bf16_n5120_k2304
