#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/kernels/bf16/bf16_a16_tma_register_b.cuh"

namespace ninfer::ops::detail {
namespace {
template <class S>
using Static = Bf16ScheduleInstance<S, 20480>;
using Gemv =
    Static<Bf16A16GemvSchedule<4, 4, 1, 8, 4, Bf16ActivationAccess::Shared,
                               Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 1, 4, 2>>;
using Short = Static<
    Bf16A16SimtSchedule<8, 8, 1, 8, 4, 2, Bf16SimtActivationAccess::WarpPacked,
                        Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 4, 1, 2, 4>>;
using Small = Static<
    Bf16A16SimtSchedule<8, 4, 1, 8, 4, 2, Bf16SimtActivationAccess::WarpPacked,
                        Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 3, 4, 1, 2, 8>>;
using S32 = Static<Bf16A16SlicedKMmaSchedule<16, 32, 16, 32, 2, Cache::cg, Cache::cg, 1, 7,
                                             Bf16MmaRaster::TokenFast>>;
using S64 = Static<Bf16A16SlicedKMmaSchedule<32, 64, 16, 16, 2, Cache::cg, Cache::cg, 1, 3,
                                             Bf16MmaRaster::TokenFast>>;
using Medium =
    Static<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 12, 3>>;
using Wide = Static<Bf16RowTailSchedule<
    Bf16A16TmaMmaSchedule<192, 128, 64, 96, 32, 2, 1, Bf16MmaRaster::TokenFast, 1, 1, 3>>>;
using Long =
    Static<Bf16A16TmaRegisterBSchedule<128, 128, 64, 64, 32, 2, Bf16MmaRaster::Grouped, 4>>;
using Bulk = Static<Bf16A16TmaMmaSchedule<128, 256, 64, 64, 64, 2, 1>>;

template <class S>
void warp(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<S>(bf16_a16_operands(x, w),
                               LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
                               Bf16MmaWarpStoreEpilogue{}, stream);
}

template <class S>
void tail(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<S>(bf16_a16_operands(x, w),
                               LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
                               Bf16TmaRowVectorEpilogue{}, stream);
}

void register_b(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    launch_bf16_a16_tma_register_b<Long>(bf16_a16_operands(x, w),
                                         LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
                                         Bf16MmaWarpStoreEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n2048_k20480(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Gemv>;
    if (tokens <= 4) return launch_bf16_simt<Short>;
    if (tokens <= 8) return launch_bf16_simt<Small>;
    if (tokens <= 64) return launch_bf16_sliced_k_mma<S32>;
    if (tokens <= 128) return launch_bf16_sliced_k_mma<S64>;
    if (tokens <= 256) return warp<Medium>;
    if (tokens <= 512) return tail<Wide>;
    if (tokens <= 1024) return register_b;
    return warp<Bulk>;
}
} // namespace ninfer::ops::detail
