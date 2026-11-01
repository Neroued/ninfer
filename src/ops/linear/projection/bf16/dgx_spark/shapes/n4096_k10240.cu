#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
template <class S>
using Static = Bf16ScheduleInstance<S, 10240>;
using Gemv =
    Static<Bf16A16GemvSchedule<4, 4, 1, 8, 4, Bf16ActivationAccess::Direct,
                               Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 1, 4, 2>>;
using Small = Static<Bf16A16SlicedKMmaSchedule<16, 16, 8, 64, 2>>;
using S32   = Static<Bf16A16SlicedKMmaSchedule<16, 32, 8, 64, 2>>;
using Hot   = Static<Bf16A16MmaSchedule<32, 64, 256, 16, 16, 2, 1, Cache::cg, Cache::cg,
                                        Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Medium =
    Static<Bf16A16TmaMmaSchedule<256, 128, 64, 128, 32, 2, 1, Bf16MmaRaster::Grouped, 4, 3>>;
using Wide  = Static<Bf16A16TmaMmaSchedule<256, 128, 64, 128, 32, 2, 1>>;
using Large = Static<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>>;
using Bulk =
    Static<Bf16A16TmaMmaSchedule<256, 128, 64, 128, 32, 2, 1, Bf16MmaRaster::Grouped, 4, 3>>;

template <class S>
void warp(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<S>(bf16_a16_operands(x, w),
                               LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
                               Bf16MmaWarpStoreEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n4096_k10240(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Gemv>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<Small>;
    if (tokens <= 32) return launch_bf16_sliced_k_mma<S32>;
    if (tokens <= 128) return launch_bf16_mma<Hot>;
    if (tokens <= 384) return warp<Medium>;
    if (tokens <= 512) return warp<Large>;
    if (tokens <= 2048) return warp<Wide>;
    return warp<Bulk>;
}
} // namespace ninfer::ops::detail
