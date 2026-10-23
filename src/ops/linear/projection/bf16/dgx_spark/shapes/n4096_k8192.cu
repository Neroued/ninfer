#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Schedule>
using Static = Bf16ScheduleInstance<Schedule, 8192>;
template <class Schedule>
using Predicated = Bf16KTailSchedule<Static<Schedule>>;
using Gemv       = Predicated<
          Bf16A16GemvSchedule<4, 4, 1, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                              Bf16PhaseOrder::RowSwizzled, 1, 1, 4, 2>>;
using Simt = Predicated<
    Bf16A16SimtSchedule<4, 1, 2, 8, 4, 2, Bf16SimtActivationAccess::WarpPacked,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2, 8>>;
using S32 = Static<Bf16A16SlicedKMmaSchedule<16, 32, 8, 64, 2>>;
using M64 = Static<Bf16A16MmaSchedule<32, 64, 256, 16, 16, 2, 1, Cache::cg, Cache::cg,
                                      Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Medium =
    Static<Bf16A16TmaMmaSchedule<256, 128, 64, 128, 32, 2, 1, Bf16MmaRaster::Grouped, 8>>;
using Window = Static<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 3, 1>>;
using T256   = Predicated<Bf16A16TmaMmaSchedule<128, 256, 64, 64, 64, 2, 1>>;
using Bulk =
    Predicated<Bf16A16TmaMmaSchedule<128, 256, 64, 64, 64, 2, 1, Bf16MmaRaster::Grouped, 12, 3>>;

template <class Schedule>
void warp(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Schedule>(
        bf16_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
        Bf16MmaWarpStoreEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n4096_k8192(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Gemv>;
    if (tokens <= 8) return launch_bf16_simt<Simt>;
    if (tokens <= 32) return launch_bf16_sliced_k_mma<S32>;
    if (tokens <= 128) return launch_bf16_mma<M64>;
    if (tokens <= 256) return warp<Medium>;
    if (tokens <= 512) return warp<Window>;
    if (tokens <= 1024) return warp<T256>;
    return warp<Bulk>;
}
} // namespace ninfer::ops::detail
