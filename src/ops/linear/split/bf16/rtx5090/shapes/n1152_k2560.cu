#include "ops/linear/split/dispatch.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear/split/bf16/launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail::linear_split {
namespace {

using S0 = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<4, 2, 1, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    2560>;
using S1 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 8, 64, 2>, 2560>;
using S2 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 16, 256, 16, 8, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    2560>;
using S3 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 32, 128, 16, 8, 3, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    2560>;
using S4 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 3>, 2560>;
using S5 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3>, 2560>;
using S6 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2>, 2560>;
using S7 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3>, 2560>;

template <class Routes>
auto select_route(int tokens, const Routes& routes) {
    if (tokens <= 1) return routes.template gemv<S0>();
    if (tokens <= 16) return routes.template sliced<S1>();
    if (tokens <= 64) return routes.template mma<S2>();
    if (tokens <= 128) return routes.template mma<S3>();
    if (tokens <= 256) return routes.template tma<S4, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 512) return routes.template tma<S5, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 1024) return routes.template tma<S6, Bf16MmaWarpStoreEpilogue>();
    return routes.template tma<S7, Bf16MmaWarpStoreEpilogue>();
}

void launch_split(const Tensor& x, Weights weights, std::span<const Tensor> out, MemoryArena*,
                  cudaStream_t stream) {
    const auto& weight = weights.first;
    using Output       = LinearBf16SegmentedOutput<512, 512, 128>;
    const Output output{{static_cast<__nv_bfloat16*>(out[0].data),
                         static_cast<__nv_bfloat16*>(out[1].data),
                         static_cast<__nv_bfloat16*>(out[2].data)}};
    select_route(x.ne[1], Bf16SplitRoutes<Output>{bf16_a16_operands(x, weight), output, stream});
}
} // namespace

const SplitShape kBf16SplitN1152K2560ThreeOutput{1152, 2560, kSplitRows<512, 512, 128>,
                                                 launch_split};

} // namespace ninfer::ops::detail::linear_split
