#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Base>
struct N640K2560 : Bf16ScheduleInstance<Base, 2560> {};

using Vector = N640K2560<
    Bf16A16GemvSchedule<4, 2, 1, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>;
template <int Tokens>
using Sliced = N640K2560<Bf16A16SlicedKMmaSchedule<16, Tokens, 4, 64, 2>>;
using Small =
    N640K2560<Bf16A16MmaSchedule<32, 16, 64, 16, 8, 3, 2, Cache::cg, Cache::cg,
                                 Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Medium = N640K2560<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 3, 2>>;
using Bulk   = N640K2560<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 2, Bf16MmaRaster::RowFast>>;

template <class S>
void warp_store(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<S>(bf16_a16_operands(x, w),
                               LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                               Bf16MmaWarpStoreEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n640_k2560(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Vector>;
    if (tokens <= 8) return launch_bf16_sliced_k_mma<Sliced<8>>;
    if (tokens <= 32) return launch_bf16_sliced_k_mma<Sliced<16>>;
    if (tokens <= 64) return launch_bf16_sliced_k_mma<Sliced<32>>;
    if (tokens <= 128) return launch_bf16_mma<Small>;
    if (tokens <= 256) return warp_store<Medium>;
    return warp_store<Bulk>;
}
} // namespace ninfer::ops::detail
