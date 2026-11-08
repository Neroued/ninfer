#include "ops/linear/bias/nvfp4_mul/dispatch.h"
#include "ops/linear/bias/nvfp4_mul/launch.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_instances.cuh"

namespace ninfer::ops::detail {
namespace {
using DecodeOne = Nvfp4ScheduleInstance<
    Nvfp4A16GemvSchedule<4, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>,
    4096>;
using Route0 = Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 8, 2>, 4096>;
using Route1 = Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 4, 2>, 4096>;
using Route2 = Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 2>, 4096>;
using Route3 = Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 1>, 4096>;
using Route4 = Nvfp4ScheduleInstance<
    Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2, Cache::cg, Cache::cg,
                        Nvfp4MmaFragmentPipeline::PingPong, true>, 4096>;
using Route5 = Nvfp4ScheduleInstance<
    Nvfp4A16MmaSchedule<64, 64, 128, 16, 32, 2, 2, Cache::cg, Cache::cg,
                        Nvfp4MmaFragmentPipeline::PingPong, true>, 4096>;
using MidRows = Nvfp4ScheduleInstance<
    Nvfp4A16MmaSchedule<64, 128, 64, 16, 64, 2, 2, Cache::cg, Cache::cg,
                        Nvfp4MmaFragmentPipeline::PingPong, true>, 4096>;
using Route6 =
    Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<128, 128, 64, 32, 64, 2, 2, Cache::cg, Cache::cg,
                                              Nvfp4MmaFragmentPipeline::PingPong, true, 0>,
                          4096>;
} // namespace

Nvfp4MulBiasRoute select_nvfp4_mul_bias_n1024_k4096(int tokens) {
    if (tokens == 1) return {nvfp4_mul_bias_gemv<DecodeOne>, false};
    if (tokens <= 8) return {nvfp4_mul_bias_sliced<Route0>, true};
    if (tokens <= 16) return {nvfp4_mul_bias_sliced<Route1>, true};
    if (tokens <= 32) return {nvfp4_mul_bias_sliced<Route2>, true};
    if (tokens <= 64) return {nvfp4_mul_bias_sliced<Route3>, true};
    if (tokens <= 96) return {nvfp4_mul_bias_mma<Route4>, true};
    if (tokens <= 192) return {nvfp4_mul_bias_mma<Route5>, true};
    if (tokens <= 384 || (tokens > 768 && tokens <= 1152))
        return {nvfp4_mul_bias_mma<MidRows>, true};
    return {nvfp4_mul_bias_mma<Route6>, true};
}
} // namespace ninfer::ops::detail
