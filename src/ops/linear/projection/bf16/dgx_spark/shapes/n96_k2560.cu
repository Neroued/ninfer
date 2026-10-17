#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear/projection/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// This geometry owns its CUDA entries even when its constants match another shape.
template <class Base>
struct Bf16N96K2560Schedule : Base {};

using S0 = Bf16N96K2560Schedule<Bf16ScheduleInstance<
    Bf16A16GemvSchedule<4, 2, 1, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    2560>>;
using S1 = Bf16N96K2560Schedule<Bf16ScheduleInstance<
    Bf16A16SimtSchedule<4, 2, 1, 8, 4, 2, Bf16SimtActivationAccess::WarpPacked,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 2, 2>,
    2560>>;
using S2 = Bf16N96K2560Schedule<
    Bf16A16SimtSchedule<8, 2, 1, 8, 4, 2, Bf16SimtActivationAccess::WarpPacked,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 2, 2>>;
using S3 =
    Bf16N96K2560Schedule<Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 8, 64, 2>, 2560>>;
using S4 =
    Bf16N96K2560Schedule<Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 32, 8, 64, 2>, 2560>>;
using S5 = Bf16N96K2560Schedule<Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 32, 256, 16, 8, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::RowFast>,
    2560>>;

template <class S>
void launch_warp(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    static_assert(!bf16_predicated_rows<S>);
    launch_bf16_a16_mma<S>(bf16_a16_operands(x, w),
                           LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                           Bf16MmaWarpStoreEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n96_k2560(std::int32_t tokens) {
    if (tokens <= 1) return launch_bf16_gemv<S0>;
    if (tokens <= 4) return launch_bf16_simt<S1>;
    if (tokens <= 16) return launch_bf16_simt<S2>;
    if (tokens <= 128) return launch_bf16_sliced_k_mma<S3>;
    if (tokens <= 512) return launch_bf16_sliced_k_mma<S4>;
    return launch_warp<S5>;
}
} // namespace ninfer::ops::detail
