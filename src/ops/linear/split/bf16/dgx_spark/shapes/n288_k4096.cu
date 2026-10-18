#include "ops/linear/split/dispatch.h"
#include "ops/linear/split/bf16/launch.cuh"

namespace ninfer::ops::detail::linear_split {
namespace {
template <class Schedule>
using Static = Bf16ScheduleInstance<Schedule, 4096>;
template <class Schedule>
using Tail = Static<Bf16KTailSchedule<Schedule>>;

using Gemv =
    Tail<Bf16A16GemvSchedule<4, 4, 1, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                             Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>;
using Short  = Tail<Bf16A16SlicedKMmaSchedule<16, 8, 16, 64, 2>>;
using Hot    = Tail<Bf16A16SlicedKMmaSchedule<16, 16, 8, 64, 2>>;
using Medium = Tail<Bf16A16SlicedKMmaSchedule<16, 32, 4, 64, 2>>;
using Mma16 =
    Static<Bf16A16MmaSchedule<48, 16, 128, 16, 8, 3, 1, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Mma32 =
    Static<Bf16A16MmaSchedule<48, 32, 256, 16, 8, 2, 1, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Tma32 = Static<Bf16A16TmaMmaSchedule<48, 32, 128, 16, 16, 2, 2, Bf16MmaRaster::RowFast>>;
using Tma64 = Static<Bf16A16TmaMmaSchedule<48, 64, 64, 16, 32, 2, 2, Bf16MmaRaster::RowFast>>;

void launch_split(const Tensor& x, Weights weights, std::span<const Tensor> out, MemoryArena*,
                  cudaStream_t stream) {
    using Output = LinearBf16SegmentedOutput<32, 128, 128>;
    const Output output{{static_cast<__nv_bfloat16*>(out[0].data),
                         static_cast<__nv_bfloat16*>(out[1].data),
                         static_cast<__nv_bfloat16*>(out[2].data)}};
    const Bf16SplitRoutes<Output> routes{bf16_a16_operands(x, weights.first), output, stream};
    const int tokens = x.ne[1];
    if (tokens == 1) return routes.gemv<Gemv>();
    if (tokens <= 16) return routes.sliced<Short>();
    if (tokens <= 32) return routes.sliced<Hot>();
    if (tokens <= 64) return routes.sliced<Medium>();
    if (tokens <= 128) return routes.mma<Mma16>();
    if (tokens <= 256) return routes.mma<Mma32>();
    if (tokens <= 512) return routes.tma<Tma32>();
    return routes.tma<Tma64>();
}
} // namespace

const SplitShape kBf16SplitN288K4096{288, 4096, kSplitRows<32, 128, 128>, launch_split};
} // namespace ninfer::ops::detail::linear_split
