#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
using S0 = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<4, 1, 1, 16, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    2560>;
using S1 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 4, 128, 2>, 2560>;
using S2 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 16, 128, 16, 16, 2, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    2560>;
using S3 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 32, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    2560>;
using S4 = Bf16A16MmaSchedule<64, 16, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using S5 = Bf16A16TmaMmaSchedule<128, 64, 128, 32, 32, 2, 1>;
using S6 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 32, 128, 16, 16, 2, 1>, 2560>;
using S7 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 48, 64, 32, 16, 2, 1>, 2560>;
using S8 = Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3, 2>;

using Wide = Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::TokenFast, 1, 3>, 2560>;
using WarpLarge = Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<128, 256, 64, 64, 64, 2, 1, Bf16MmaRaster::Grouped, 8, 2>, 2560>;

template <class Base>
struct Bf16N13952K2560Schedule : Base {};

template <class Base>
void vector_projection(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Bf16N13952K2560Schedule<Base>>(
        bf16_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        Bf16TmaVectorEpilogue{}, stream);
}

void warp_large(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Bf16N13952K2560Schedule<WarpLarge>>(
        bf16_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        Bf16MmaWarpStoreEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n13952_k2560(std::int32_t tokens) {
    if (tokens <= 1) return launch_bf16_gemv<Bf16N13952K2560Schedule<S0>>;
    if (tokens <= 8) return launch_bf16_sliced_k_mma<Bf16N13952K2560Schedule<S1>>;
    if (tokens <= 16) return launch_bf16_mma<Bf16N13952K2560Schedule<S2>>;
    if (tokens <= 32) return launch_bf16_mma<Bf16N13952K2560Schedule<S3>>;
    if (tokens <= 48) return launch_bf16_mma<Bf16N13952K2560Schedule<S4>>;
    if (tokens <= 64) return launch_bf16_tma_mma<Bf16N13952K2560Schedule<S5>>;
    if (tokens <= 96) return launch_bf16_tma_mma<Bf16N13952K2560Schedule<S6>>;
    if (tokens <= 128) return launch_bf16_tma_mma<Bf16N13952K2560Schedule<S7>>;
    if (tokens <= 256) return launch_bf16_tma_mma<Bf16N13952K2560Schedule<S8>>;
    if (tokens > 3840) return warp_large;
    if (tokens > 384 && tokens <= 512) return launch_bf16_tma_mma<Bf16N13952K2560Schedule<Wide>>;
    return vector_projection<Wide>;
}
} // namespace ninfer::ops::detail
