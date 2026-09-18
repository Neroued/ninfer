#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_ksplit_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Q8N6144K5120;
using Access   = Q8KSplitScaleAccess;
using Stage    = Q8KSplitActivationStage;
using C4 = Q8KSplitSchedule<16, 8, 2, Access::Direct, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C8 = Q8KSplitSchedule<8, 8, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
// Bound registers to allow three resident 256-thread CTAs per SM.
using C16 = Q8KSplitSchedule<8, 16, 3, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C24 = Q8KSplitSchedule<4, 24, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C32 = Q8KSplitSchedule<4, 32, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C40 = Q8KSplitSchedule<4, 40, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C48 = Q8KSplitSchedule<4, 48, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C56 = Q8KSplitSchedule<4, 56, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
} // namespace

Q8Launch select_q8_n6144_k5120(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_ksplit<Geometry, 4, C4>;
    if (tokens <= 8) return launch_q8_ksplit<Geometry, 8, C8>;
    if (tokens <= 16) return launch_q8_ksplit<Geometry, 16, C16>;
    if (tokens <= 24) return launch_q8_ksplit<Geometry, 24, C24>;
    if (tokens <= 32) return launch_q8_ksplit<Geometry, 32, C32>;
    if (tokens <= 40) return launch_q8_ksplit<Geometry, 40, C40>;
    if (tokens <= 48) return launch_q8_ksplit<Geometry, 48, C48>;
    if (tokens <= 56) return launch_q8_ksplit<Geometry, 56, C56>;
    if (tokens <= 192) return launch_q8_mma_r32_c64;
    return launch_q8_mma_r64_c128;
}

} // namespace ninfer::ops::detail
