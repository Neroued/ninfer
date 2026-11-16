#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
template <class S>
using Static = Bf16ScheduleInstance<S, 2048>;
using Gemv =
    Static<Bf16A16GemvSchedule<8, 4, 1, 8, 4, Bf16ActivationAccess::Direct,
                               Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 1, 4, 2>>;
using Small  = Static<Bf16A16SlicedKMmaSchedule<16, 32, 16, 32, 2, Cache::cg, Cache::cg, 1, 7,
                                                Bf16MmaRaster::TokenFast>>;
using Medium = Static<Bf16A16SlicedKMmaSchedule<64, 32, 8, 32, 2, Cache::cg, Cache::cg, 1, 7,
                                                Bf16MmaRaster::TokenFast>>;
using Wide =
    Static<Bf16A16TmaMmaSchedule<64, 64, 128, 32, 16, 3, 1, Bf16MmaRaster::TokenFast, 1, 1, 3>>;
using Bulk = Static<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>>;

template <class S>
void warp(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<S>(bf16_a16_operands(x, w),
                               LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
                               Bf16MmaWarpStoreEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n4096_k2048(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Gemv>;
    if (tokens <= 32) return launch_bf16_sliced_k_mma<Small>;
    if (tokens <= 96) return launch_bf16_sliced_k_mma<Medium>;
    if (tokens <= 128) return warp<Wide>;
    return warp<Bulk>;
}
} // namespace ninfer::ops::detail
