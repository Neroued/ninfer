#pragma once

#include "ops/common/cuda/memory.cuh"

#include "ops/linear/kernels/q8/q8_instances.cuh"

namespace ninfer::ops::detail::q8_n12288_k2560 {

using Gemv      = Q8A16GemvSchedule<8, 1, 2, 2560, 1>;
using Small     = Q8A16SlicedKMmaSchedule<16, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 2560>;
using Medium    = Q8A16MmaSchedule<32, 32, 128, 16, 16, 2, 3>;
using Group32   = Q8A16GroupedSlicedKMmaSchedule<32, 2, 2, 1, 1, 2560, Cache::cg, Cache::ca, true>;
using Mid1      = Q8A16SlicedKMmaSchedule<40, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 2560>;
using Mid2      = Q8A16SlicedKMmaSchedule<40, 4, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 2560>;
using Mma80     = Q8A16MmaSchedule<32, 80, 64, 32, 16, 2, 3>;
using Mma80Full = Q8A16MmaSchedule<48, 80, 64, 48, 16, 2, 3>;
using Mma96     = Q8A16MmaSchedule<48, 96, 64, 48, 16, 2, 2>;
using Mma112    = Q8A16MmaSchedule<48, 112, 64, 48, 16, 2, 2>;
using Mma112Serial = Q8A16MmaSchedule<48, 112, 64, 16, 16, 2, 2, Q8MmaFragmentPipeline::Serial>;
using Mma128       = Q8A16MmaSchedule<48, 128, 64, 16, 16, 2, 2, Q8MmaFragmentPipeline::Serial>;
using Bulk = Q8A16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Q8MmaFragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Cache::ca, false, 2560, 20>;

template <class Routes>
auto select(int tokens, const Routes& routes) {
    // Full token tiles select different kernels from tails; retain their measured winners.
    // The 34-column single-stage sliced schedule also has a repeatable advantage.
    if (tokens == 1) return routes.template gemv<Gemv>();
    if (tokens <= 16) return routes.template sliced<Small>();
    if (tokens == 32) return routes.template grouped<Group32>();
    if (tokens < 32) return routes.template mma<Medium>();
    if (tokens == 34) return routes.template sliced<Mid1>();
    if (tokens <= 40) return routes.template sliced<Mid2>();
    if (tokens == 64) return routes.template mma<q8_instances::MmaR48T64>();
    if (tokens < 64) return routes.template mma<q8_instances::MmaR32T64>();
    if (tokens == 80) return routes.template mma<Mma80Full>();
    if (tokens < 80) return routes.template mma<Mma80>();
    if (tokens <= 96) return routes.template mma<Mma96>();
    if (tokens >= 104 && tokens <= 110) return routes.template mma<Mma112Serial>();
    if (tokens <= 112) return routes.template mma<Mma112>();
    if (tokens <= 128) return routes.template mma<Mma128>();
    return routes.template mma<Bulk>();
}

} // namespace ninfer::ops::detail::q8_n12288_k2560
