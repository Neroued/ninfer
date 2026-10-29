#include "ops/linear/split/dispatch.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear/split/q8/dgx_spark/launch.cuh"

namespace ninfer::ops::detail::linear_split {
namespace {

using Gemv    = Q8A16GemvSchedule<4, 2, 2, 2560, 4>;
using Small8  = Q8A16SlicedKMmaSchedule<8, 8, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                        Q8ActivationStage::PaddedZero, 2560>;
using Small16 = Q8A16SlicedKMmaSchedule<16, 8, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                        Q8ActivationStage::PaddedZero, 2560>;
using Small32 = Q8A16SlicedKMmaSchedule<32, 8, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                        Q8ActivationStage::PaddedZero, 2560>;
using Small40 = Q8A16SlicedKMmaSchedule<40, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                        Q8ActivationStage::PaddedZero, 2560>;
using Hot32   = Q8A16TmaMmaSchedule<64, 32, 32, 16, 3, 8, 2560>;
using Hot64   = Q8A16TmaMmaSchedule<64, 64, 64, 16, 3, 8, 2560>;
using Hot96   = Q8A16TmaMmaSchedule<64, 96, 64, 16, 2, 8, 2560>;
using Bulk128 = Q8A16TmaMmaSchedule<64, 128, 64, 16, 2, 8, 2560>;

using Output = LinearBf16SegmentedOutput<5120, 3072>;

struct Routes : Q8SplitRoutes<Output> {
    void full() const { launch_q8_n8192_k2560_full(operands, output, stream); }
};

void select_route(int tokens, const Routes& routes) {
    if (tokens == 1) return routes.gemv<Gemv>();
    if (tokens <= 8) return routes.sliced<Small8>();
    if (tokens <= 16) return routes.sliced<Small16>();
    if (tokens <= 32) return routes.sliced<Small32>();
    if (tokens <= 40) return routes.sliced<Small40>();
    if (tokens <= 96) return routes.tma<Hot32>();
    if (tokens <= 128) return routes.tma<Hot64>();
    if (tokens <= 192) return routes.tma<Hot96>();
    if (tokens >= 1024 && tokens % 128 == 0) return routes.full();
    return routes.tma<Bulk128>();
}

void launch_split(const Tensor& x, Weights weights, std::span<const Tensor> out, MemoryArena*,
                  cudaStream_t stream) {
    const Output output{
        {static_cast<__nv_bfloat16*>(out[0].data), static_cast<__nv_bfloat16*>(out[1].data)}};
    select_route(x.ne[1], Routes{{{q8_linear_operands(x, weights.first), output, stream}}});
}
} // namespace

const SplitShape kQ8SplitN8192K2560{8192, 2560, kSplitRows<5120, 3072>, launch_split};

} // namespace ninfer::ops::detail::linear_split
