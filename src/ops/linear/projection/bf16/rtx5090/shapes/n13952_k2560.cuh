#pragma once

#include "ops/common/cuda/memory.cuh"

#include "ops/linear/kernels/bf16/bf16_schedule.cuh"

namespace ninfer::ops::detail::bf16_n13952_k2560 {

using S0 = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<4, 1, 1, 16, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    2560>;
using S1 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 4, 128, 2>, 2560>;
using S2 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 16, 128, 16, 16, 2, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    2560>;
using S3 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 32, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    2560>;
using S4 = Bf16A16MmaSchedule<64, 16, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using S5 = Bf16A16TmaMmaSchedule<128, 64, 128, 32, 32, 2, 1>;
using S6 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 32, 128, 16, 16, 2, 1>, 2560>;
using S7 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 48, 64, 32, 16, 2, 1>, 2560>;
using S8 = Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3, 2>;
using S9 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 3, 1>, 2560>;

template <class Routes>
auto select(int tokens, const Routes& routes) {
    if (tokens <= 1) return routes.template gemv<S0>();
    if (tokens <= 8) return routes.template sliced<S1>();
    if (tokens <= 16) return routes.template mma<S2>();
    if (tokens <= 32) return routes.template mma<S3>();
    if (tokens <= 48) return routes.template mma<S4>();
    if (tokens <= 64) return routes.template tma<S5>();
    if (tokens <= 96) return routes.template tma<S6>();
    if (tokens <= 128) return routes.template tma<S7>();
    if (tokens <= 512) return routes.template tma<S8>();
    return routes.template tma<S9>();
}

} // namespace ninfer::ops::detail::bf16_n13952_k2560
