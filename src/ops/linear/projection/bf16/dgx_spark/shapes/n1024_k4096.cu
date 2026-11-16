#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
template <class S>
using Static = Bf16ScheduleInstance<S, 4096>;
using Gemv =
    Static<Bf16A16GemvSchedule<16, 2, 1, 8, 4, Bf16ActivationAccess::Direct,
                               Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 2, 2>>;
using S8     = Static<Bf16A16SlicedKMmaSchedule<16, 8, 16, 32, 2, Cache::cg, Cache::cg, 1, 7,
                                                Bf16MmaRaster::TokenFast>>;
using S16    = Static<Bf16A16SlicedKMmaSchedule<16, 16, 16, 32, 2, Cache::cg, Cache::cg, 1, 7,
                                                Bf16MmaRaster::TokenFast>>;
using S32    = Static<Bf16A16SlicedKMmaSchedule<64, 32, 8, 32, 2, Cache::cg, Cache::cg, 1, 7,
                                                Bf16MmaRaster::TokenFast>>;
using Medium = Static<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>>;
using Mid    = Static<Bf16A16TmaMmaSchedule<128, 64, 64, 64, 16, 3, 1>>;
using Large  = Static<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>>;
using Bulk   = Static<Bf16A16TmaMmaSchedule<128, 256, 64, 64, 64, 2, 1, Bf16MmaRaster::Grouped, 8>>;
using Balanced =
    Static<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 12, 3>>;
using Wide =
    Static<Bf16A16TmaMmaSchedule<256, 128, 64, 128, 32, 2, 1, Bf16MmaRaster::Grouped, 4, 3>>;

template <class S>
void warp(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<S>(bf16_a16_operands(x, w),
                               LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
                               Bf16MmaWarpStoreEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n1024_k4096(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Gemv>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<S8>;
    if (tokens <= 32) return launch_bf16_sliced_k_mma<S16>;
    if (tokens <= 128) return launch_bf16_sliced_k_mma<S32>;
    if (tokens <= 256) return warp<Medium>;
    if (tokens <= 384) return warp<Mid>;
    if (tokens <= 768) return warp<Large>;
    if (tokens <= 1536) return warp<Bulk>;
    if (tokens <= 2304) return warp<Balanced>;
    return warp<Wide>;
}
} // namespace ninfer::ops::detail
