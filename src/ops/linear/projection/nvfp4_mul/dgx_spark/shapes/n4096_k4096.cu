#include "ops/linear/projection/nvfp4_mul/dispatch.h"
#include "ops/linear/projection/nvfp4_mul/launch.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_instances.cuh"

namespace ninfer::ops::detail {
namespace {
using Route0 = Nvfp4ScheduleInstance<
    Nvfp4A16SimtSchedule<4, 2, 2, 16, 2, 4, Nvfp4SimtActivationAccess::TokenPacked,
                         Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 1,
                         Nvfp4SimtBlockOrder::RowsContiguous, 2>,
    4096, 2>;
using Route1 = Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 16, 1>, 4096>;
using Route2 = Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 4, 2>, 4096>;
using Route3 = Nvfp4ScheduleInstance<
    Nvfp4A16MmaSchedule<64, 32, 128, 16, 32, 2, 2, Cache::cg, Cache::cg,
                        Nvfp4MmaFragmentPipeline::PingPong, true, 0, true>, 4096>;
using Route4 = Nvfp4ScheduleInstance<
    Nvfp4A16MmaSchedule<64, 24, 128, 16, 24, 2, 2, Cache::cg, Cache::cg,
                        Nvfp4MmaFragmentPipeline::PingPong, true, 0, true>, 4096>;
using Route5 = Nvfp4ScheduleInstance<
    Nvfp4A16MmaSchedule<64, 96, 64, 16, 96, 2, 2, Cache::cg, Cache::cg,
                        Nvfp4MmaFragmentPipeline::PingPong, true, 0, true>, 4096>;
using Route6 = Nvfp4ScheduleInstance<
    Nvfp4A16MmaSchedule<64, 64, 64, 16, 64, 2, 2, Cache::cg, Cache::cg,
                        Nvfp4MmaFragmentPipeline::PingPong, true, 0, true>, 4096>;
using Route7 = Nvfp4ScheduleInstance<
    Nvfp4A16MmaSchedule<128, 128, 64, 32, 64, 2, 2, Cache::cg, Cache::cg,
                        Nvfp4MmaFragmentPipeline::PingPong, true>, 4096>;
using Route8 = Nvfp4ScheduleInstance<
    Nvfp4A16MmaSchedule<128, 128, 64, 16, 128, 2, 2, Cache::cg, Cache::cg,
                        Nvfp4MmaFragmentPipeline::PingPong, true, 0, true>, 4096>;
} // namespace

Nvfp4MulRoute select_nvfp4_mul_n4096_k4096(int tokens) {
    if (tokens <= 2) return {nvfp4_mul_linear_simt<Route0>, false};
    if (tokens <= 8) return {nvfp4_mul_linear_sliced<Route1>, true};
    if (tokens <= 16) return {nvfp4_mul_linear_sliced<Route2>, true};
    if (tokens <= 64) return {nvfp4_mul_linear_mma<Route3>, true};
    // Three 24-column tiles cover 65..72 tokens without the fourth shared
    // memory residency slot being lost to a larger input tile.
    if (tokens <= 72) return {nvfp4_mul_linear_mma<Route4>, true};
    if (tokens <= 96) return {nvfp4_mul_linear_mma<Route5>, true};
    if (tokens <= 256) return {nvfp4_mul_linear_mma<Route6>, true};
    if (tokens <= 512) return {nvfp4_mul_linear_mma<Route7>, true};
    return {nvfp4_mul_linear_mma<Route8>, true};
}
} // namespace ninfer::ops::detail
