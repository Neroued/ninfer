#include "ops/linear/split/dispatch.h"
#include "ops/linear/kernels/fp8/native_a16.cuh"
#include "ops/linear/kernels/fp8/native_a16_sliced_k.cuh"
#include "ops/linear/kernels/fp8_block/operands.h"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/kernels/bf16/bf16_a16_tma_register_b.cuh"
#include "ops/linear/split/bf16/launch.cuh"

namespace ninfer::ops::detail::linear_split {
namespace {
template <class S>
using Fp8 = NativeFp8ScheduleInstance<S, 1536>;
template <class S>
using Bf16 = Bf16ScheduleInstance<S, 1536>;

template <class S>
struct RegisterWeights : S {
    static constexpr bool kRegisterWeights = true;
};

using Fp8Short  = Fp8<NativeFp8A16SlicedKSchedule<16, 8, 8, 128, 1>>;
using Fp8Small  = Fp8<NativeFp8A16SlicedKSchedule<16, 16, 2, 128, 2>>;
using Fp8Medium = Fp8<NativeFp8MmaSchedule<32, 32, 128, 16, 16, 2>>;
using Fp8Bulk   = RegisterWeights<Fp8<NativeFp8MmaSchedule<128, 64, 128, 32, 32, 1, 2>>>;
using Bf16Gemv =
    Bf16<Bf16A16GemvSchedule<4, 1, 2, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                             Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>;
using Bf16Short = Bf16<Bf16A16SlicedKMmaSchedule<16, 8, 8, 64, 1>>;
using Bf16Small = Bf16<Bf16A16SlicedKMmaSchedule<16, 16, 8, 64, 1>>;
using Bf16Medium =
    Bf16<Bf16A16MmaSchedule<32, 32, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                            Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::RowFast>>;
using Bf16Large  = Bf16<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3, 2, Bf16MmaRaster::RowFast>>;
using Bf16Larger = Bf16<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 2, Bf16MmaRaster::RowFast>>;
using Bf16Bulk = Bf16<Bf16A16TmaRegisterBSchedule<128, 128, 64, 64, 32, 2, Bf16MmaRaster::RowFast>>;

void launch_first(const Tensor& x, const Weight& weight, LinearBf16Output out,
                  cudaStream_t stream) {
    const auto p = fp8_block_a16_operands(x, weight);
    if (p.tokens <= 16)
        return launch_native_fp8_a16_sliced_k<Fp8Short>(p, out, LinearIdentityEpilogue{}, stream);
    if (p.tokens <= 32)
        return launch_native_fp8_a16_sliced_k<Fp8Small>(p, out, LinearIdentityEpilogue{}, stream);
    if (p.tokens <= 64)
        return launch_native_fp8_a16_mma<Fp8Medium>(p, out, LinearIdentityEpilogue{}, stream);
    if (p.tokens <= 256)
        return launch_native_fp8_a16_mma<Fp8Bulk>(p, out, LinearIdentityEpilogue{}, stream);
    launch_native_fp8_a16_mma<Fp8Bulk>(p, out, Bf16MmaStreamingWarpStoreEpilogue{}, stream);
}

void launch_second(const Bf16SplitRoutes<LinearBf16Output>& routes) {
    const int tokens = routes.operands.tokens;
    if (tokens == 1) return routes.gemv<Bf16Gemv>();
    if (tokens <= 8) return routes.sliced<Bf16Short>();
    if (tokens <= 16) return routes.sliced<Bf16Small>();
    if (tokens <= 32) return routes.mma<Bf16Medium>();
    if (tokens <= 64) return routes.tma<Bf16Large, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 128) return routes.tma<Bf16Larger, Bf16MmaWarpStoreEpilogue>();
    launch_bf16_a16_tma_register_b<Bf16Bulk>(routes.operands, routes.output,
                                             Bf16MmaWarpStoreEpilogue{}, routes.stream);
}

void launch(const Tensor& x, Weights weights, std::span<const Tensor> outputs, MemoryArena*,
            cudaStream_t stream) {
    launch_first(x, weights.first, {static_cast<__nv_bfloat16*>(outputs[0].data), 16384}, stream);
    launch_second({bf16_a16_operands(x, *weights.second),
                   {static_cast<__nv_bfloat16*>(outputs[1].data), 4096},
                   stream});
}
} // namespace

const SplitShape kFp8BlockBf16SplitN16384N4096K1536{16384,  1536,    kSplitRows<16384, 4096>,
                                                    launch, nullptr, 4096};
} // namespace ninfer::ops::detail::linear_split
