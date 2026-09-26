#pragma once
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"

namespace ninfer::ops::detail {
template <int Tokens, int Warps, int Stages>
using Nvfp4SlicedInstance =
    Nvfp4A16SlicedKMmaSchedule<Warps, Tokens,
                               (Nvfp4A16SlicedKMmaSchedule<Warps, Tokens, 1, Cache::ca, Cache::cg,
                                                           Nvfp4ActivationStage::PaddedZero,
                                                           Stages>::kSharedBytes <= 99 * 1024 / 2
                                    ? 2
                                    : 1),
                               Cache::ca, Cache::cg, Nvfp4ActivationStage::PaddedZero, Stages>;
} // namespace ninfer::ops::detail
