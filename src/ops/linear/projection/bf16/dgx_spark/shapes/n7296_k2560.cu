#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Base>
struct N7296K2560 : Base {};

template <class Base>
using Schedule = N7296K2560<Bf16ScheduleInstance<Base, 2560>>;
using Vector =
    Schedule<Bf16A16GemvSchedule<4, 2, 1, 8, 4, Bf16ActivationAccess::Direct,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>;
using Sliced = Schedule<Bf16A16SlicedKMmaSchedule<16, 8, 4, 128, 2>>;
template <int Stages>
using Small =
    Schedule<Bf16A16MmaSchedule<32, 32, 128, 16, 8, Stages, 1, Cache::cg, Cache::cg,
                                Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Bulk = Schedule<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3>>;
using Wide = Schedule<Bf16A16TmaMmaSchedule<128, 256, 64, 64, 64, 2>>;

template <class S>
void warp_store(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<S>(bf16_a16_operands(x, weight),
                               LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
                               Bf16MmaWarpStoreEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n7296_k2560(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Vector>;
    if (tokens <= 8) return launch_bf16_sliced_k_mma<Sliced>;
    if (tokens <= 64) return launch_bf16_mma<Small<3>>;
    if (tokens <= 96) return launch_bf16_mma<Small<2>>;
    if (tokens <= 128) return launch_bf16_mma<Small<3>>;
    if (tokens <= 512) return warp_store<Bulk>;
    if (tokens <= 1024) return warp_store<Wide>;
    return warp_store<Bulk>;
}
} // namespace ninfer::ops::detail
