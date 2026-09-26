#pragma once
#include "ops/linear/q8/q8_geometry.h"
#include "ops/linear/q8/q8_schedule.cuh"

namespace ninfer::ops::detail {
template <int TileTokens, int ActiveTokens>
using Q8SlicedKDefault = Q8A16SlicedKMmaSchedule<
    TileTokens, 8, 1, TileTokens == 8 ? 5 : (TileTokens == 16 ? 4 : (TileTokens == 24 ? 3 : 2)),
    (ActiveTokens > 4 ? Q8ScaleAccess::Shared : Q8ScaleAccess::Direct)>;

namespace q8_instances {
using MmaR32T64  = Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>;
using MmaR32T96  = Q8A16MmaSchedule<32, 96, 64, 32, 16, 2, 2>;
using MmaR32T128 = Q8A16MmaSchedule<32, 128, 64, 32, 16, 2, 2>;
using MmaR48T64  = Q8A16MmaSchedule<48, 64, 64, 48, 16, 2, 3>;
using MmaR64T96  = Q8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>;
// The wide route of the linear family is the one schedule measured to want cg on the predicated
// path; every other schedule, here and in the six other Ops that instantiate this same tile, keeps
// the inherited ca.
using MmaR64T128 = Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2, Q8MmaFragmentPipeline::PingPong,
                                    Cache::cg, Cache::cg, Cache::cg>;
using MmaR96T96  = Q8A16MmaSchedule<96, 96, 64, 48, 16, 2, 2>;
using MmaR128T64 = Q8A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>;
using MmaR128T80 = Q8A16MmaSchedule<128, 80, 64, 64, 16, 2, 2>;
// K128 single-activation-stage schedules shared with fused consumers.
using MmaR64x16T48K128A1 = Q8A16MmaSchedule<64, 48, 128, 16, 24, 1, 2>;
using MmaR64x32T64K128A1 = Q8A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>;

using SimtR8T4         = Q8A16SimtSchedule<8, 4, 1, 32, 2, Cache::cg, 1>;
using SimtR8T8         = Q8A16SimtSchedule<8, 8, 1, 32, 2, Cache::cg, 1>;
using GemvR4W1K16384   = Q8A16GemvSchedule<4, 1, 2, 16384>;
using SimtR4T4W2G16S2  = Q8A16SimtSchedule<4, 4, 2, 16, 2, Cache::cg, 1>;
using SlicedR16T16W8S2 = Q8A16SlicedKMmaSchedule<16, 8, 2, 1, Q8ScaleAccess::Shared>;
using SlicedR16T16W4S2 = Q8A16SlicedKMmaSchedule<16, 4, 2, 1, Q8ScaleAccess::Shared, Cache::ca,
                                                 Cache::cg, Q8ActivationStage::PaddedZero>;
using SlicedR16T32W4S2 = Q8A16SlicedKMmaSchedule<32, 4, 2, 1, Q8ScaleAccess::Shared>;
} // namespace q8_instances
} // namespace ninfer::ops::detail
