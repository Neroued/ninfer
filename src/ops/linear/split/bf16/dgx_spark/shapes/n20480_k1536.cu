#include "ops/linear/split/dispatch.h"
#include "ops/linear/split/bf16/launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail::linear_split {
namespace {
template <class S>
using Static = Bf16ScheduleInstance<S, 1536>;
template <class S>
using Predicated = Bf16KTailSchedule<Static<S>>;
using S16        = Predicated<Bf16A16SlicedKMmaSchedule<16, 16, 8, 64, 2>>;
using S32        = Static<Bf16A16SlicedKMmaSchedule<16, 32, 8, 64, 2>>;
using M64        = Static<Bf16A16MmaSchedule<16, 64, 256, 16, 16, 2, 1, Cache::cg, Cache::cg,
                                             Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using M32 =
    Predicated<Bf16A16MmaSchedule<32, 32, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                                  Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using T64  = Predicated<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>>;
using T128 = Predicated<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>>;

template <class Output>
void select_route(int tokens, const Bf16SplitRoutes<Output>& routes) {
    if (tokens <= 16) return routes.template sliced<S16>();
    if (tokens <= 32) return routes.template sliced<S32>();
    if (tokens <= 64) return routes.template mma<M64>();
    if (tokens <= 128) return routes.template mma<M32>();
    if (tokens <= 256) return routes.template tma<T64>();
    return routes.template tma<T128, Bf16MmaWarpStoreEpilogue>();
}

void launch_split(const Tensor& x, Weights weights, std::span<const Tensor> outputs, MemoryArena*,
                  cudaStream_t stream) {
    using Output = LinearBf16SegmentedOutput<16384, 4096>;
    const Output output{{static_cast<__nv_bfloat16*>(outputs[0].data),
                         static_cast<__nv_bfloat16*>(outputs[1].data)}};
    select_route(x.ne[1],
                 Bf16SplitRoutes<Output>{bf16_a16_operands(x, weights.first), output, stream});
}
} // namespace

const SplitShape kBf16SplitN20480K1536{20480, 1536, kSplitRows<16384, 4096>, launch_split};
} // namespace ninfer::ops::detail::linear_split
