#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear/projection/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Base>
struct Bf16N2560K2560Schedule : Base {};

using Gemv = Bf16N2560K2560Schedule<Bf16ScheduleInstance<
    Bf16A16GemvSchedule<4, 1, 2, 16, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    2560>>;
using Sliced =
    Bf16N2560K2560Schedule<Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 4, 64, 3>, 2560>>;
using Hot64      = Bf16N2560K2560Schedule<Bf16ScheduleInstance<
         Bf16A16MmaSchedule<64, 32, 128, 32, 16, 2, 2, Cache::cg, Cache::cg,
                            Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
         2560>>;
using Prefill64  = Bf16N2560K2560Schedule<Bf16ScheduleInstance<
     Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 2, Bf16MmaRaster::TokenFast, 1>, 2560>>;
using Prefill128 = Bf16N2560K2560Schedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, 2560>>;
// Reuse input fragments across rows with different pipeline depths in the hot bands.
using HotTwoStage   = Bf16N2560K2560Schedule<Bf16ScheduleInstance<
      Bf16A16MmaSchedule<32, 32, 128, 16, 8, 2, 1, Cache::cg, Cache::cg,
                         Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
      2560>>;
using HotThreeStage = Bf16N2560K2560Schedule<Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 32, 128, 16, 8, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    2560>>;
using Prefill192    = Bf16N2560K2560Schedule<
       Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<160, 192, 64, 80, 48, 2, 1>, 2560>>;
using LargeGrouped = Bf16N2560K2560Schedule<Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<256, 128, 64, 128, 32, 2, 1, Bf16MmaRaster::Grouped, 6, 3>, 2560>>;

template <class S>
void launch_warp_projection(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<S>(bf16_a16_operands(x, w),
                               LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                               Bf16MmaWarpStoreEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n2560_k2560(std::int32_t tokens) {
    if (tokens <= 1) return launch_bf16_gemv<Gemv>;
    if (tokens <= 8) return launch_bf16_sliced_k_mma<Sliced>;
    if (tokens <= 64) return launch_bf16_mma<HotThreeStage>;
    if (tokens <= 96) return launch_bf16_mma<HotTwoStage>;
    if (tokens <= 128) return launch_bf16_mma<Hot64>;
    if (tokens <= 256) return launch_warp_projection<Prefill64>;
    // 160x192 fits one CTA wave through T576. Its next wave is slower;
    // the measured second favorable band starts after T896.
    if (tokens <= 576 || (tokens > 896 && tokens <= 1152))
        return launch_warp_projection<Prefill192>;
    if (tokens <= 2048) return launch_warp_projection<Prefill128>;
    return launch_warp_projection<LargeGrouped>;
}
} // namespace ninfer::ops::detail
