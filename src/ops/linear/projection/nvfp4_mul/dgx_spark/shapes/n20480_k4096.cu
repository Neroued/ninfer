#include "ops/linear/projection/nvfp4_mul/dispatch.h"
#include "ops/linear/projection/nvfp4_mul/launch.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_instances.cuh"

namespace ninfer::ops::detail {
namespace {
using Route0 = Nvfp4ScheduleInstance<
    Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>,
    4096>;
using DecodePair = Nvfp4ScheduleInstance<
    Nvfp4A16SimtSchedule<4, 2, 2, 16, 2, 4, Nvfp4SimtActivationAccess::TokenPacked,
                         Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 1,
                         Nvfp4SimtBlockOrder::RowsContiguous, 2>,
    4096, 2>;
using Route1 = Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 16, 1>, 4096>;
using Route2 = Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 8, 2>, 4096>;
using Route3 = Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 2>, 4096>;
using Route4 = Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 64, 128, 16, 32, 2, 2, Cache::cg, Cache::cg,
                        Nvfp4MmaFragmentPipeline::PingPong, true, 0, true>, 4096>;
using Route5 = Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 128, 64, 16, 64, 2, 2, Cache::cg, Cache::cg,
                        Nvfp4MmaFragmentPipeline::PingPong, true, 0, true>, 4096>;
using Route6 =
    Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<128, 128, 64, 32, 64, 2, 2, Cache::cg, Cache::cg,
                                              Nvfp4MmaFragmentPipeline::PingPong, true, 8>,
                          4096>;
} // namespace

Nvfp4MulRoute select_nvfp4_mul_n20480_k4096(int tokens) {
    if (tokens <= 1) return {nvfp4_mul_linear_gemv<Route0>, false};
    if (tokens <= 2) return {nvfp4_mul_linear_simt<DecodePair>, false};
    if (tokens <= 4) return {nvfp4_mul_linear_sliced<Route1>, true};
    if (tokens <= 16) return {nvfp4_mul_linear_sliced<Route2>, true};
    if (tokens <= 32) return {nvfp4_mul_linear_sliced<Route3>, true};
    if (tokens <= 64) return {nvfp4_mul_linear_mma<Route4>, true};
    if (tokens <= 128) return {nvfp4_mul_linear_mma<Route5>, true};
    return {nvfp4_mul_linear_mma<Route6>, true};
}
} // namespace ninfer::ops::detail
