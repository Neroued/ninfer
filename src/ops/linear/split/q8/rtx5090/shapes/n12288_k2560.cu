#include "ops/linear/split/dispatch.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear/kernels/q8/q8_output_launch.cuh"

namespace ninfer::ops::detail::linear_split {
namespace {

using Mma64     = Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>;
using Mma64Full = Q8A16MmaSchedule<48, 64, 64, 48, 16, 2, 3>;
using Gemv      = Q8A16GemvSchedule<8, 1, 2, 2560, 1>;
using Small     = Q8A16SlicedKMmaSchedule<16, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 2560>;
using Medium    = Q8A16MmaSchedule<32, 32, 128, 16, 16, 2, 3>;
using Group32   = Q8A16GroupedSlicedKMmaSchedule<32, 2, 2, 1, 1, 2560, Cache::cg, Cache::ca, true>;
using Mid1      = Q8A16SlicedKMmaSchedule<40, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 2560>;
using Mid2      = Q8A16SlicedKMmaSchedule<40, 4, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 2560>;
using Mma80     = Q8A16MmaSchedule<32, 80, 64, 32, 16, 2, 3>;
using Mma80Full = Q8A16MmaSchedule<48, 80, 64, 48, 16, 2, 3>;
using Mma96     = Q8A16MmaSchedule<48, 96, 64, 48, 16, 2, 2>;
using Mma112    = Q8A16MmaSchedule<48, 112, 64, 48, 16, 2, 2>;
using Mma112Serial = Q8A16MmaSchedule<48, 112, 64, 16, 16, 2, 2, Q8MmaFragmentPipeline::Serial>;
using Mma128       = Q8A16MmaSchedule<48, 128, 64, 16, 16, 2, 2, Q8MmaFragmentPipeline::Serial>;
using Bulk = Q8A16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Q8MmaFragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Cache::ca, false, 2560, 20>;

template <class Routes>
auto select_route(int tokens, const Routes& routes) {
    // Full token tiles select different kernels from tails; retain their measured winners.
    // The 34-column single-stage sliced schedule also has a repeatable advantage.
    if (tokens == 1) return routes.template gemv<Gemv>();
    if (tokens <= 16) return routes.template sliced<Small>();
    if (tokens == 32) return routes.template grouped<Group32>();
    if (tokens < 32) return routes.template mma<Medium>();
    if (tokens == 34) return routes.template sliced<Mid1>();
    if (tokens <= 40) return routes.template sliced<Mid2>();
    if (tokens == 64) return routes.template mma<Mma64Full>();
    if (tokens < 64) return routes.template mma<Mma64>();
    if (tokens == 80) return routes.template mma<Mma80Full>();
    if (tokens < 80) return routes.template mma<Mma80>();
    if (tokens <= 96) return routes.template mma<Mma96>();
    if (tokens >= 104 && tokens <= 110) return routes.template mma<Mma112Serial>();
    if (tokens <= 112) return routes.template mma<Mma112>();
    if (tokens <= 128) return routes.template mma<Mma128>();
    return routes.template mma<Bulk>();
}

void launch_split(const Tensor& x, Weights weights, std::span<const Tensor> out, MemoryArena*,
                  cudaStream_t stream) {
    const auto& weight = weights.first;
    using Output       = LinearBf16SegmentedOutput<6144, 6144>;
    const Output output{
        {static_cast<__nv_bfloat16*>(out[0].data), static_cast<__nv_bfloat16*>(out[1].data)}};
    select_route(x.ne[1], Q8A16OutputLaunch<Output>{q8_linear_operands(x, weight), output, stream});
}
} // namespace

const SplitShape kQ8SplitN12288K2560{12288, 2560, kSplitRows<6144, 6144>, launch_split};

} // namespace ninfer::ops::detail::linear_split
