#pragma once

#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/kernels/bf16/bf16_schedule.cuh"

namespace ninfer::ops::detail::bf16_n32_k2304 {

template <class Schedule>
using StaticK = Bf16ScheduleInstance<Schedule, 2304>;

// Eight warps cooperate on one output. The final 256 values occupy only one warp.
using DecodeSimt = StaticK<Bf16KTailSchedule<
    Bf16A16SimtSchedule<8, 8, 1, 8, 4, 1, Bf16SimtActivationAccess::DirectStream,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2, 1>>>;
using Sliced16 = StaticK<Bf16A16SlicedKMmaSchedule<16, 16, 12, 32, 2, Cache::cg, Cache::cg, 1, 1>>;
using Tma128 = StaticK<Bf16A16TmaMmaSchedule<32, 128, 64, 16, 32, 3, 1, Bf16MmaRaster::TokenFast>>;

} // namespace ninfer::ops::detail::bf16_n32_k2304
