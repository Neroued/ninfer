#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear/projection/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// Keep each original schedule and K-tail policy local to this shape.
template <class Base>
struct Bf16HcUpSchedule : Base {};

using Original0 = Bf16KTailSchedule<Bf16ScheduleInstance<
    Bf16A16GemvSchedule<2, 1, 1, 16, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    320>>;
using Original1 = Bf16A16MmaSchedule<32, 8, 64, 16, 8, 3, 2, Cache::cg, Cache::cg,
                                     Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using Original2 = Bf16A16MmaSchedule<64, 16, 64, 16, 8, 3, 2, Cache::cg, Cache::cg,
                                     Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using Original3 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 3, 2>, 320>;
using S0        = Bf16HcUpSchedule<Original0>;
using S1        = Bf16HcUpSchedule<Original1>;
using S2        = Bf16HcUpSchedule<Original2>;
using S3        = Bf16HcUpSchedule<Original3>;
using Hot =
    Bf16HcUpSchedule<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 2, 2>, 320>>;
using Prefill64 =
    Bf16HcUpSchedule<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 2>, 320>>;
using Prefill128 = Bf16HcUpSchedule<Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2, 2, Bf16MmaRaster::Grouped, 8, 3>, 320>>;

template <class Schedule>
void launch_vector(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Schedule>(bf16_a16_operands(x, w),
                                      LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                      Bf16TmaVectorEpilogue{}, stream);
}

template <class S>
void launch_hot_warp(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    static_assert(!bf16_predicated_rows<S>);
    launch_bf16_a16_tma_mma<S>(bf16_a16_operands(x, w),
                               LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                               Bf16MmaWarpStoreEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n10240_k320(std::int32_t tokens) {
    if (tokens <= 1) return launch_bf16_gemv<S0>;
    if (tokens <= 16) return launch_bf16_mma<S1>;
    if (tokens <= 48) return launch_bf16_mma<S2>;
    if (tokens <= 64) return launch_hot_warp<S3>;
    if (tokens <= 128) return launch_hot_warp<Hot>;
    // The wider token tile loses on odd 64-token tails in the measured low-T band.
    const auto token_tiles64 = (static_cast<std::int64_t>(tokens) + 63) / 64;
    if (tokens <= 448 || (tokens <= 896 && token_tiles64 % 2 != 0)) return launch_vector<Prefill64>;
    return launch_vector<Prefill128>;
}
} // namespace ninfer::ops::detail
