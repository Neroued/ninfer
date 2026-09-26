#pragma once
#include "ops/linear/fp8/fp8_schedule.cuh"

namespace ninfer::ops::detail {
using Fp8A8T64R128K128 =
    Fp8A8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Fp8A8T16R64K128 = Fp8A8MmaSchedule<16, 64, 128, 1, 2, 2, 4, Cache::cg, Cache::cg,
                                         Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Fp8A8T16R128K128 =
    Fp8A8MmaSchedule<16, 128, 128, 1, 4, 2, 2, Cache::cg, Cache::cg,
                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Fp8A8T32R32K128 = Fp8A8MmaSchedule<32, 32, 128, 1, 2, 3, 3, Cache::cg, Cache::cg,
                                         Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Fp8A8T32R64K128 = Fp8A8MmaSchedule<32, 64, 128, 1, 2, 3, 2, Cache::cg, Cache::cg,
                                         Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Fp8A8T32R128K128 =
    Fp8A8MmaSchedule<32, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Fp8A8T64R64K128 = Fp8A8MmaSchedule<64, 64, 128, 2, 2, 3, 2, Cache::cg, Cache::cg,
                                         Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Fp8A8T64R64K128S2 =
    Fp8A8MmaSchedule<64, 64, 128, 2, 2, 2, 3, Cache::cg, Cache::cg,
                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Fp8A8T64R128K256 =
    Fp8A8MmaSchedule<64, 128, 256, 2, 4, 2, 1, Cache::cg, Cache::cg,
                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
template <int Tokens, int Warps, int Stages>
using Fp8SlicedInstance = Fp8A16SlicedKMmaSchedule<
    Warps, Tokens, (Stages * (16 * Warps * 64 + Warps * Tokens * 128) > 48 * 1024 ? 1 : 2),
    Cache::ca, Cache::cg, Fp8ActivationStage::PaddedZero, Stages>;
} // namespace ninfer::ops::detail
