#include "ops/linear/split/dispatch.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear/split/q8/dgx_spark/launch.cuh"

namespace ninfer::ops::detail::linear_split {
namespace {
template <class Base>
struct SplitSchedule : Base {};

using Hot96   = SplitSchedule<Q8A16TmaMmaSchedule<64, 96, 64, 16, 2, 8, 2560>>;
using Bulk128 = SplitSchedule<Q8A16TmaMmaSchedule<64, 128, 64, 16, 2, 8, 2560>>;

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

using Output = LinearBf16SegmentedOutput<6144, 6144>;

struct Routes : Q8SplitRoutes<Output> {
    void full() const { launch_q8_n12288_k2560_full(operands, output, stream); }
};

template <class Routes>
auto select_route(int tokens, const Routes& routes) {
    if (tokens == 1) return routes.template gemv<Gemv>();
    if (tokens <= 16) return routes.template sliced<Small>();
    if (tokens == 32) return routes.template grouped<Group32>();
    if (tokens < 32) return routes.template mma<Medium>();
    if (tokens == 34) return routes.template sliced<Mid1>();
    if (tokens <= 40) return routes.template sliced<Mid2>();
    if (tokens == 64) return routes.template mma<Mma64Full>();
    if (tokens < 64) return routes.template mma<Mma64>();
    if (tokens >= 1024 && tokens % 128 == 0) return routes.full();
    if (tokens <= 96 || (tokens > 128 && tokens <= 192)) return routes.template tma<Hot96>();
    return routes.template tma<Bulk128>();
}

void launch_split(const Tensor& x, Weights weights, std::span<const Tensor> out, MemoryArena*,
                  cudaStream_t stream) {
    const auto& weight = weights.first;
    using Output       = LinearBf16SegmentedOutput<6144, 6144>;
    const Output output{
        {static_cast<__nv_bfloat16*>(out[0].data), static_cast<__nv_bfloat16*>(out[1].data)}};
    select_route(x.ne[1], Routes{{{q8_linear_operands(x, weight), output, stream}}});
}
} // namespace

const SplitShape kQ8SplitN12288K2560{12288, 2560, kSplitRows<6144, 6144>, launch_split};

} // namespace ninfer::ops::detail::linear_split
