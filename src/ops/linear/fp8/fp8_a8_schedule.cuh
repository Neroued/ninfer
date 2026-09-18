#pragma once
#include "ops/linear/fp8/fp8_a8_mma.cuh"

namespace ninfer::ops::detail {
using Fp8A8DefaultSchedule =
    Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg, Fp8MmaFragmentPipeline::PingPong,
                   Fp8MmaRaster::TokenFast>;
}
