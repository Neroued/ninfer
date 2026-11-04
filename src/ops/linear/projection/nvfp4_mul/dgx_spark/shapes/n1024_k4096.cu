#include "ops/linear/projection/nvfp4_mul/dispatch.h"
#include "ops/linear/projection/nvfp4_mul/launch.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_instances.cuh"

namespace ninfer::ops::detail {
namespace {
using Route0 = Nvfp4ScheduleInstance<
    Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>,
    4096>;
using Route1 = Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 8, 2>, 4096>;
using Route2 = Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 8, 2>, 4096>;
using Route3 = Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 4, 2>, 4096>;
using Route4 = Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 1>, 4096>;
using Route5 = Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>, 4096>;
using Route6 = Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>, 4096>;
using Route7 =
    Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<128, 128, 64, 64, 32, 2, 2, Cache::cg, Cache::cg,
                                              Nvfp4MmaFragmentPipeline::PingPong, true, 0>,
                          4096>;
} // namespace

Nvfp4MulRoute select_nvfp4_mul_n1024_k4096(int tokens) {
    if (tokens <= 1) return {nvfp4_mul_linear_gemv<Route0>, false};
    if (tokens <= 8) return {nvfp4_mul_linear_sliced<Route1>, true};
    if (tokens <= 16) return {nvfp4_mul_linear_sliced<Route2>, true};
    if (tokens <= 32) return {nvfp4_mul_linear_sliced<Route3>, true};
    if (tokens <= 64) return {nvfp4_mul_linear_sliced<Route4>, true};
    if (tokens <= 128) return {nvfp4_mul_linear_mma<Route5>, true};
    if (tokens <= 256) return {nvfp4_mul_linear_mma<Route6>, true};
    return {nvfp4_mul_linear_mma<Route7>, true};
}
} // namespace ninfer::ops::detail
