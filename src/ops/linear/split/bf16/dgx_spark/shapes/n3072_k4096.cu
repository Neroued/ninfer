#include "ops/linear/split/dispatch.h"
#include "ops/linear/split/bf16/launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail::linear_split {
namespace {
template <class S>
using Static = Bf16ScheduleInstance<S, 4096>;
using Gemv =
    Static<Bf16A16GemvSchedule<8, 4, 1, 8, 4, Bf16ActivationAccess::Direct,
                               Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 1, 4, 2>>;
using Small  = Static<Bf16A16SlicedKMmaSchedule<16, 32, 16, 32, 2, Cache::cg, Cache::cg, 1, 7,
                                                Bf16MmaRaster::TokenFast>>;
using Medium = Static<Bf16A16SlicedKMmaSchedule<32, 64, 8, 32, 2, Cache::cg, Cache::cg, 1, 7,
                                                Bf16MmaRaster::TokenFast>>;
using Bulk   = Static<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>>;

void launch(const Tensor& x, Weights weights, std::span<const Tensor> outputs, MemoryArena*,
            cudaStream_t stream) {
    using Output = LinearBf16SegmentedOutput<512, 512, 2048>;
    const Output out{{static_cast<__nv_bfloat16*>(outputs[0].data),
                      static_cast<__nv_bfloat16*>(outputs[1].data),
                      static_cast<__nv_bfloat16*>(outputs[2].data)}};
    const Bf16SplitRoutes<Output> routes{bf16_a16_operands(x, weights.first), out, stream};
    if (x.ne[1] == 1) return routes.gemv<Gemv>();
    if (x.ne[1] <= 64) return routes.sliced<Small>();
    if (x.ne[1] <= 128) return routes.sliced<Medium>();
    return routes.tma<Bulk, Bf16MmaWarpStoreEpilogue>();
}
} // namespace

const SplitShape kBf16SplitN3072K4096{3072, 4096, kSplitRows<512, 512, 2048>, launch};
} // namespace ninfer::ops::detail::linear_split
