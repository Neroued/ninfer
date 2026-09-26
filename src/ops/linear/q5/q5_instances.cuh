#pragma once
#include "ops/linear/q5/q5_schedule.cuh"

namespace ninfer::ops::detail::q5_instances {

using MmaR64T128 = Q5A16MmaSchedule<64, 128, 64, 64, 32, 2, 1, Q5MmaFragmentPipeline::Serial,
                                    Cache::cg, Cache::cg, Q5ScaleLoad::Pair32>;

using MmaR32T128 = Q5A16MmaSchedule<32, 128, 64, 32, 32, 2, 1, Q5MmaFragmentPipeline::Serial,
                                    Cache::cg, Cache::cg, Q5ScaleLoad::Pair32>;

using GemvR16W1G16S2XK5120 = Q5A16GemvSchedule<16, 1, 16, 2, Cache::ca, 1, true, 5120>;
using DirectR1T1W4K5120    = Q5A16DirectSimtSchedule<1, 1, 4, 4, 10, 5120, true>;
using DirectR1T2W4K5120    = Q5A16DirectSimtSchedule<1, 2, 4, 4, 10, 5120, true>;
using DirectR1T3W4K5120    = Q5A16DirectSimtSchedule<1, 3, 4, 4, 10, 5120, true>;

using DirectR1T1W4K6144 = Q5A16DirectSimtSchedule<1, 1, 4, 4, 10, 6144, true>;
using DirectR1T2W2K6144 = Q5A16DirectSimtSchedule<1, 2, 2, 8, 16, 6144, true>;
using DirectR1T3W2K6144 = Q5A16DirectSimtSchedule<1, 3, 2, 8, 16, 6144, true>;

using DirectR1T1W4K17408 = Q5A16DirectSimtSchedule<1, 1, 4, 4, 10, 17408, true>;
using DirectR1T2W2K17408 = Q5A16DirectSimtSchedule<1, 2, 2, 8, 16, 17408, true>;
using DirectR1T3W2K17408 = Q5A16DirectSimtSchedule<1, 3, 2, 8, 16, 17408, true>;
using DirectR1T4W2K17408 = Q5A16DirectSimtSchedule<1, 4, 2, 8, 16, 17408, true>;

using DirectR2T4W2G8B4         = Q5A16DirectSimtSchedule<2, 4, 2, 8, 4>;
using SlicedR16T8W4S2          = Q5A16SlicedKMmaSchedule<16, 8, 4, 2, Cache::cg, Cache::ca, 2>;
using SlicedR16T16W4S2         = Q5A16SlicedKMmaSchedule<16, 16, 4, 2, Cache::cg, Cache::ca, 2>;
using SlicedR16T32W4S2         = Q5A16SlicedKMmaSchedule<16, 32, 4, 2, Cache::cg, Cache::ca, 2>;
using SlicedR32T24W4S2Pairwise = Q5A16SlicedKMmaSchedule<32, 24, 4, 2, Cache::cg, Cache::ca, 2, 0,
                                                         24, Q5SlicedKReduction::Pairwise>;
using SlicedR32T32W4S2         = Q5A16SlicedKMmaSchedule<32, 32, 4, 2, Cache::cg, Cache::ca, 2>;
using SlicedR32T32W4S1         = Q5A16SlicedKMmaSchedule<32, 32, 4, 1, Cache::cg, Cache::ca, 2>;
using SlicedR32T64W2S2         = Q5A16SlicedKMmaSchedule<32, 64, 2, 2, Cache::cg, Cache::ca, 2>;
using SlicedR32T64W2S1         = Q5A16SlicedKMmaSchedule<32, 64, 2, 1, Cache::cg, Cache::ca, 2>;
using SlicedR16T8Capacity4 = Q5A16SlicedKMmaSchedule<16, 8, 4, 2, Cache::cg, Cache::ca, 2, 0, 4>;
using SlicedR16T24W4S2     = Q5A16SlicedKMmaSchedule<16, 24, 4, 2, Cache::cg, Cache::ca, 2>;
using SlicedR32T16W4S2     = Q5A16SlicedKMmaSchedule<32, 16, 4, 2, Cache::cg, Cache::ca, 2>;
using MmaR64T96K128S1A1 = Q5A16MmaSchedule<64, 96, 128, 32, 32, 1, 2, Q5MmaFragmentPipeline::Serial,
                                           Cache::cg, Cache::cg, Q5ScaleLoad::Pair32, 1>;
using SlicedR16T16W2S2  = Q5A16SlicedKMmaSchedule<16, 16, 2, 2, Cache::cg, Cache::ca, 2>;
using SlicedR32T32W2S2  = Q5A16SlicedKMmaSchedule<32, 32, 2, 2, Cache::cg, Cache::ca, 2>;
using DirectR2T4W4G4B4  = Q5A16DirectSimtSchedule<2, 4, 4, 4, 4>;
using MmaR32T32K128S2A2 = Q5A16MmaSchedule<32, 32, 128, 16, 16, 2, 2, Q5MmaFragmentPipeline::Serial,
                                           Cache::cg, Cache::cg, Q5ScaleLoad::Pair32, 2>;
} // namespace ninfer::ops::detail::q5_instances
