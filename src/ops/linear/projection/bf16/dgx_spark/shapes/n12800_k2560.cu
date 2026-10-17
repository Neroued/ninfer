#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Base>
struct Bf16PleKvSchedule : Base {};

using S0 = Bf16PleKvSchedule<Bf16ScheduleInstance<
    Bf16A16GemvSchedule<2, 1, 1, 16, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    2560>>;
using S1 = Bf16PleKvSchedule<Bf16ScheduleInstance<
    Bf16A16MmaSchedule<64, 8, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    2560>>;
using S2 = Bf16PleKvSchedule<Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 16, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    2560>>;
using S3 = Bf16PleKvSchedule<Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 32, 128, 16, 16, 2, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    2560>>;
using S4 = Bf16PleKvSchedule<
    Bf16A16MmaSchedule<64, 16, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using S5 = Bf16PleKvSchedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 64, 128, 32, 32, 2, 1>, 2560>>;
using S6 =
    Bf16PleKvSchedule<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 32, 64, 32, 16, 3, 1>, 2560>>;
using S8 =
    Bf16PleKvSchedule<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 3, 1>, 2560>>;
using S9 = Bf16PleKvSchedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, 2560>>;
using Hot =
    Bf16PleKvSchedule<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 3, 1>, 2560>>;
using PrefillVector = Bf16PleKvSchedule<Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::TokenFast, 1, 3>, 2560>>;
using WarpLarge     = Bf16PleKvSchedule<Bf16ScheduleInstance<
        Bf16A16TmaMmaSchedule<256, 128, 64, 128, 32, 2, 1, Bf16MmaRaster::Grouped, 5, 1>, 2560>>;

void launch_prefill_vector(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<PrefillVector>(
        bf16_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        Bf16TmaVectorEpilogue{}, stream);
}

void launch_warp_large(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<WarpLarge>(bf16_a16_operands(x, w),
                                       LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                       Bf16MmaWarpStoreEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n12800_k2560(std::int32_t tokens) {
    if (tokens <= 1) return launch_bf16_gemv<S0>;
    if (tokens <= 8) return launch_bf16_mma<S1>;
    if (tokens <= 16) return launch_bf16_mma<S2>;
    if (tokens <= 32) return launch_bf16_mma<S3>;
    if (tokens <= 48) return launch_bf16_mma<S4>;
    if (tokens <= 64) return launch_bf16_tma_mma<S5>;
    if (tokens <= 96) return launch_bf16_tma_mma<S6>;
    if (tokens <= 128) return launch_bf16_tma_mma<Hot>;
    if (tokens <= 384) return launch_bf16_tma_mma<S8>;
    if (tokens <= 512) return launch_bf16_tma_mma<S9>;
    if (tokens <= 3840) return launch_prefill_vector;
    // Late wave/cache controls favor the old identity path through T4032.
    if (tokens <= 4032) return launch_bf16_tma_mma<S9>;
    return launch_warp_large;
}
} // namespace ninfer::ops::detail
