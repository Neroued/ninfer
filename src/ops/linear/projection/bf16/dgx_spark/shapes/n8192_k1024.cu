#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
template <class S>
using Static = Bf16ScheduleInstance<S, 1024>;
using Gemv =
    Static<Bf16A16GemvSchedule<8, 1, 1, 8, 4, Bf16ActivationAccess::Direct,
                               Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 2, 2>>;
using Small  = Static<Bf16A16SlicedKMmaSchedule<16, 16, 4, 64, 1>>;
using Hot    = Static<Bf16A16MmaSchedule<32, 32, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                                         Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using M32    = Static<Bf16A16MmaSchedule<32, 32, 256, 32, 8, 2, 1, Cache::cg, Cache::cg,
                                         Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Medium = Static<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 2, 1>>;
using Large  = Static<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>>;
using Bulk =
    Static<Bf16A16TmaMmaSchedule<256, 128, 64, 128, 32, 2, 1, Bf16MmaRaster::Grouped, 4, 3>>;

template <class S>
void warp(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<S>(bf16_a16_operands(x, w),
                               LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
                               Bf16MmaWarpStoreEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n8192_k1024(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Gemv>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<Small>;
    if (tokens <= 32) return launch_bf16_mma<M32>;
    if (tokens <= 128) return launch_bf16_mma<Hot>;
    if (tokens <= 256) return warp<Medium>;
    if (tokens <= 1024) return warp<Large>;
    return warp<Bulk>;
}
} // namespace ninfer::ops::detail
