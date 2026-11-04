#include "ops/linear/projection/nvfp4_mul/dispatch.h"
#include "ops/linear/projection/nvfp4_mul/launch.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_instances.cuh"

namespace ninfer::ops::detail {
namespace {
using Route0 = Nvfp4ScheduleInstance<
    Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Streaming, 2>,
    10240>;
using Route1 = Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 16, 1>, 10240>;
using Route2 = Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 8, 2>, 10240>;
using Route3 = Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>, 10240>;
using Route4 =
    Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<128, 128, 64, 64, 32, 2, 2, Cache::cg, Cache::cg,
                                              Nvfp4MmaFragmentPipeline::PingPong, true, 8>,
                          10240>;
} // namespace

Nvfp4MulRoute select_nvfp4_mul_n4096_k10240(int tokens) {
    if (tokens <= 1) return {nvfp4_mul_linear_gemv<Route0>, false};
    if (tokens <= 8) return {nvfp4_mul_linear_sliced<Route1>, true};
    if (tokens <= 16) return {nvfp4_mul_linear_sliced<Route2>, true};
    if (tokens <= 64) return {nvfp4_mul_linear_mma<Route3>, true};
    return {nvfp4_mul_linear_mma<Route4>, true};
}
} // namespace ninfer::ops::detail
