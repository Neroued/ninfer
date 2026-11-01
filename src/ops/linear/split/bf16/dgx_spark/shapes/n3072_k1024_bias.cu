#include "ops/linear/split/dispatch.h"
#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail::linear_split {
namespace {
template <class S>
using Static = Bf16ScheduleInstance<S, 1024>;
using Gemv =
    Static<Bf16A16GemvSchedule<4, 4, 1, 8, 4, Bf16ActivationAccess::Direct,
                               Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>;
using S8   = Static<Bf16A16SlicedKMmaSchedule<16, 8, 8, 64, 1>>;
using S16  = Static<Bf16A16SlicedKMmaSchedule<16, 16, 8, 64, 2>>;
using T32  = Static<Bf16A16TmaMmaSchedule<64, 32, 64, 32, 16, 3, 2>>;
using T64  = Static<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>>;
using M32  = Static<Bf16A16MmaSchedule<32, 32, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using T128 = Static<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>>;
using T256 = Static<Bf16A16TmaMmaSchedule<128, 256, 64, 64, 64, 2, 1>>;
using Bulk =
    Static<Bf16A16TmaMmaSchedule<256, 128, 64, 128, 32, 2, 1, Bf16MmaRaster::Grouped, 4, 3>>;
} // namespace

void launch_bf16_bias_n3072_k1024(const Tensor& x, const Weight& weight, const Tensor& bias,
                                  std::span<const Tensor> outputs, cudaStream_t stream) {
    const auto p = bf16_a16_operands(x, weight);
    const LinearBf16SegmentedOutput<1024, 1024, 1024> out{
        {static_cast<__nv_bfloat16*>(outputs[0].data), static_cast<__nv_bfloat16*>(outputs[1].data),
         static_cast<__nv_bfloat16*>(outputs[2].data)}};
    const LinearBiasEpilogue scalar{static_cast<const __nv_bfloat16*>(bias.data)};
    const Bf16BiasWarpStoreEpilogue warp{{}, scalar.bias};
    if (p.tokens == 1) return launch_bf16_a16_gemv<Gemv>(p, out, scalar, stream);
    if (p.tokens <= 8) return launch_bf16_a16_sliced_k_mma<S8>(p, out, scalar, stream);
    if (p.tokens <= 16) return launch_bf16_a16_sliced_k_mma<S16>(p, out, scalar, stream);
    if (p.tokens <= 32) return launch_bf16_a16_tma_mma<T32>(p, out, scalar, stream);
    if (p.tokens <= 64) return launch_bf16_a16_tma_mma<T64>(p, out, warp, stream);
    if (p.tokens <= 128) return launch_bf16_a16_mma<M32>(p, out, scalar, stream);
    if (p.tokens <= 256) return launch_bf16_a16_tma_mma<T128>(p, out, warp, stream);
    if (p.tokens <= 2048) return launch_bf16_a16_tma_mma<T256>(p, out, warp, stream);
    launch_bf16_a16_tma_mma<Bulk>(p, out, warp, stream);
}

} // namespace ninfer::ops::detail::linear_split
