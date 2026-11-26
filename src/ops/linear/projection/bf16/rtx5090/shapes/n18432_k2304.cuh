#pragma once

#include "ops/linear/kernels/bf16/bf16_schedule.cuh"

namespace ninfer::ops::detail::bf16_n18432_k2304 {

template <class Schedule>
using StaticK = Bf16ScheduleInstance<Schedule, 2304>;

using Sliced16 = StaticK<Bf16A16SlicedKMmaSchedule<64, 16, 4, 32, 2>>;
using Tma32 =
    StaticK<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 3, 2, Bf16MmaRaster::TokenFast, 1, 4>>;
using Tma48 =
    StaticK<Bf16A16TmaMmaSchedule<128, 48, 64, 32, 16, 2, 1, Bf16MmaRaster::TokenFast, 1, 4>>;
using Tma64 =
    StaticK<Bf16A16TmaMmaSchedule<128, 64, 64, 64, 32, 2, 1, Bf16MmaRaster::TokenFast, 1, 4>>;
using Tma80 =
    StaticK<Bf16A16TmaMmaSchedule<128, 80, 64, 32, 16, 3, 1, Bf16MmaRaster::Grouped, 4, 2>>;
using Tma64Narrow =
    StaticK<Bf16A16TmaMmaSchedule<96, 64, 64, 48, 16, 2, 2, Bf16MmaRaster::Grouped, 8, 4>>;
using Tma64Grouped =
    StaticK<Bf16A16TmaMmaSchedule<128, 64, 64, 64, 32, 2, 1, Bf16MmaRaster::Grouped, 8, 4>>;

template <class Routes>
auto select(int tokens, const Routes& routes) {
    if (tokens <= 16) return routes.template sliced<Sliced16>();
    if (tokens <= 64) return routes.template tma<Tma32>();
    if (tokens <= 96) return routes.template tma<Tma48>();
    if (tokens <= 128) return routes.template tma<Tma64>();
    if (tokens <= 144) return routes.template tma<Tma48>();
    if (tokens <= 160) return routes.template tma<Tma80>();
    if (tokens <= 384) return routes.template tma<Tma64Narrow>();
    return routes.template tma<Tma64Grouped>();
}

} // namespace ninfer::ops::detail::bf16_n18432_k2304
