#include "ops/linear/bias/nvfp4_mul/dispatch.h"
#include "ops/linear/bias/nvfp4_mul/launch.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_instances.cuh"

namespace ninfer::ops::detail {
namespace {
using Route0 = Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 8, 2>, 4096>;
using Route1 = Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 4, 2>, 4096>;
using Route2 = Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 4, 2>, 4096>;
using Route3 = Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 1>, 4096>;
using Route4 = Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>, 4096>;
using Route5 = Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>, 4096>;
using Route6 =
    Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<128, 128, 64, 64, 32, 2, 2, Cache::cg, Cache::cg,
                                              Nvfp4MmaFragmentPipeline::PingPong, true, 0>,
                          4096>;
} // namespace

Nvfp4MulBiasLaunch select_nvfp4_mul_bias_n1024_k4096(int tokens) {
    if (tokens <= 8) return nvfp4_mul_bias_sliced<Route0>;
    if (tokens <= 16) return nvfp4_mul_bias_sliced<Route1>;
    if (tokens <= 32) return nvfp4_mul_bias_sliced<Route2>;
    if (tokens <= 64) return nvfp4_mul_bias_sliced<Route3>;
    if (tokens <= 128) return nvfp4_mul_bias_mma<Route4>;
    if (tokens <= 256) return nvfp4_mul_bias_mma<Route5>;
    return nvfp4_mul_bias_mma<Route6>;
}
} // namespace ninfer::ops::detail
