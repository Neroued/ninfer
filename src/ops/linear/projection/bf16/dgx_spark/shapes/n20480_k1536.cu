#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Schedule>
using Static = Bf16ScheduleInstance<Schedule, 1536>;
template <class Schedule>
using Predicated = Bf16KTailSchedule<Static<Schedule>>;
using S16        = Predicated<Bf16A16SlicedKMmaSchedule<16, 16, 8, 64, 2>>;
using S32        = Static<Bf16A16SlicedKMmaSchedule<16, 32, 8, 64, 2>>;
using M64        = Static<Bf16A16MmaSchedule<16, 64, 256, 16, 16, 2, 1, Cache::cg, Cache::cg,
                                             Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using M32 =
    Predicated<Bf16A16MmaSchedule<32, 32, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                                  Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using T64  = Predicated<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>>;
using T128 = Predicated<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>>;

template <class Schedule>
void warp(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Schedule>(
        bf16_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
        Bf16MmaWarpStoreEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n20480_k1536(std::int32_t tokens) {
    if (tokens <= 16) return launch_bf16_sliced_k_mma<S16>;
    if (tokens <= 32) return launch_bf16_sliced_k_mma<S32>;
    if (tokens <= 64) return launch_bf16_mma<M64>;
    if (tokens <= 128) return launch_bf16_mma<M32>;
    if (tokens <= 256) return launch_bf16_tma_mma<T64>;
    return warp<T128>;
}
} // namespace ninfer::ops::detail
