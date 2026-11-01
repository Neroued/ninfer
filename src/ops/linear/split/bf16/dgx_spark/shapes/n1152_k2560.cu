#include "ops/linear/split/dispatch.h"
#include "ops/linear/split/bf16/launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail::linear_split {
namespace {
template <class Base>
struct SplitN1152K2560 : Base {};

template <class Base>
using Schedule = SplitN1152K2560<Bf16ScheduleInstance<Base, 2560>>;
using Vector =
    Schedule<Bf16A16GemvSchedule<4, 2, 1, 8, 4, Bf16ActivationAccess::Direct,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>;
using Sliced = Schedule<Bf16A16SlicedKMmaSchedule<16, 8, 4, 128, 2>>;
template <int Stages>
using Small =
    Schedule<Bf16A16MmaSchedule<32, 32, 128, 16, 8, Stages, 1, Cache::cg, Cache::cg,
                                Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Medium = Schedule<Bf16A16TmaMmaSchedule<128, 80, 64, 32, 16, 3>>;
using Bulk   = Schedule<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3>>;
using BulkRows =
    Schedule<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::RowFast>>;
using Wide = Schedule<Bf16A16TmaMmaSchedule<128, 256, 64, 64, 64, 2, 1, Bf16MmaRaster::RowFast>>;
using Tall = Schedule<Bf16A16TmaMmaSchedule<192, 128, 64, 64, 32, 2>>;

template <class Output>
void select_route(int tokens, const Bf16SplitRoutes<Output>& routes) {
    if (tokens == 1) return routes.template gemv<Vector>();
    if (tokens <= 8) return routes.template sliced<Sliced>();
    if (tokens <= 64) return routes.template mma<Small<3>>();
    if (tokens <= 128) return routes.template mma<Small<2>>();
    if (tokens <= 256) return routes.template tma<Medium, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 512) return routes.template tma<Bulk, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 1024) return routes.template tma<Tall>();
    if (tokens <= 2048) return routes.template tma<BulkRows, Bf16MmaWarpStoreEpilogue>();
    return routes.template tma<Wide, Bf16MmaWarpStoreEpilogue>();
}

void launch_split(const Tensor& x, Weights weights, std::span<const Tensor> out, MemoryArena*,
                  cudaStream_t stream) {
    using Output = LinearBf16SegmentedOutput<256, 256, 512, 128>;
    const Output output{
        {static_cast<__nv_bfloat16*>(out[0].data), static_cast<__nv_bfloat16*>(out[1].data),
         static_cast<__nv_bfloat16*>(out[2].data), static_cast<__nv_bfloat16*>(out[3].data)}};
    select_route(x.ne[1],
                 Bf16SplitRoutes<Output>{bf16_a16_operands(x, weights.first), output, stream});
}

void launch_three_output(const Tensor& x, Weights weights, std::span<const Tensor> out,
                         MemoryArena*, cudaStream_t stream) {
    using Output = LinearBf16SegmentedOutput<512, 512, 128>;
    const Output output{{static_cast<__nv_bfloat16*>(out[0].data),
                         static_cast<__nv_bfloat16*>(out[1].data),
                         static_cast<__nv_bfloat16*>(out[2].data)}};
    select_route(x.ne[1],
                 Bf16SplitRoutes<Output>{bf16_a16_operands(x, weights.first), output, stream});
}
} // namespace

const SplitShape kBf16SplitN1152K2560{1152, 2560, kSplitRows<256, 256, 512, 128>, launch_split};
const SplitShape kBf16SplitN1152K2560ThreeOutput{1152, 2560, kSplitRows<512, 512, 128>,
                                                 launch_three_output};
} // namespace ninfer::ops::detail::linear_split
