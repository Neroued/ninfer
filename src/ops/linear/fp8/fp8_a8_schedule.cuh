#pragma once

#include "ops/linear/fp8/fp8_a8_mma.cuh"
#include "ops/linear/fp8/fp8_config.h"

namespace ninfer::ops::detail {

template <class Geometry>
struct Fp8LinearA8ProductionSchedule;

template <>
struct Fp8LinearA8ProductionSchedule<Fp8AttnInputGeometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8GdnInputGeometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8MlpGateUpGeometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8Residual6144Geometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8Residual17408Geometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

} // namespace ninfer::ops::detail
