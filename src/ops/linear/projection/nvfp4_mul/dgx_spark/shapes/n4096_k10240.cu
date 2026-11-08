#include "ops/linear/projection/nvfp4_mul/dispatch.h"
#include "ops/linear/projection/nvfp4_mul/launch.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_instances.cuh"

namespace ninfer::ops::detail {
namespace {
using Route0 = Nvfp4ScheduleInstance<
    Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Streaming, 2>,
    10240>;
using Route1 = Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 8, 2>, 10240>;
using Route2 = Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 8, 2>, 10240>;
using Route3 = Nvfp4ScheduleInstance<
    Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2, Cache::cg, Cache::cg,
                        Nvfp4MmaFragmentPipeline::PingPong, true, 0, true>, 10240>;
using Route4 = Nvfp4ScheduleInstance<
    Nvfp4A16MmaSchedule<64, 32, 128, 16, 32, 2, 2, Cache::cg, Cache::cg,
                        Nvfp4MmaFragmentPipeline::PingPong, true, 8, true>, 10240>;
using Route5 = Nvfp4ScheduleInstance<
    Nvfp4A16MmaSchedule<64, 64, 128, 16, 32, 2, 2, Cache::cg, Cache::cg,
                        Nvfp4MmaFragmentPipeline::PingPong, true, 8, true>, 10240>;
using Route6 = Nvfp4ScheduleInstance<
    Nvfp4A16MmaSchedule<64, 96, 64, 16, 96, 2, 2, Cache::cg, Cache::cg,
                        Nvfp4MmaFragmentPipeline::PingPong, true, 8, true>, 10240>;
using Route7 = Nvfp4ScheduleInstance<
    Nvfp4A16MmaSchedule<64, 128, 64, 16, 64, 2, 2, Cache::cg, Cache::cg,
                        Nvfp4MmaFragmentPipeline::PingPong, true, 8, true>, 10240>;
using Route8 = Nvfp4ScheduleInstance<
    Nvfp4A16MmaSchedule<128, 128, 64, 32, 64, 2, 2, Cache::cg, Cache::cg,
                        Nvfp4MmaFragmentPipeline::PingPong, true, 8>, 10240>;
} // namespace

Nvfp4MulRoute select_nvfp4_mul_n4096_k10240(int tokens) {
    if (tokens <= 1) return {nvfp4_mul_linear_gemv<Route0>, false};
    if (tokens <= 8) return {nvfp4_mul_linear_sliced<Route1>, true};
    if (tokens <= 16) return {nvfp4_mul_linear_sliced<Route2>, true};
    if (tokens <= 32) return {nvfp4_mul_linear_mma<Route3>, true};
    if (tokens <= 96) return {nvfp4_mul_linear_mma<Route4>, true};
    if (tokens <= 128) return {nvfp4_mul_linear_mma<Route5>, true};
    if (tokens <= 192) return {nvfp4_mul_linear_mma<Route6>, true};
    if (tokens <= 256) return {nvfp4_mul_linear_mma<Route7>, true};
    return {nvfp4_mul_linear_mma<Route8>, true};
}
} // namespace ninfer::ops::detail
