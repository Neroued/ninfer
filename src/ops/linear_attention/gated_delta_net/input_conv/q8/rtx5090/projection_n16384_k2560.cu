#include "ops/linear_attention/gated_delta_net/input_conv/q8/launch.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear/kernels/q8/q8_output_launch.cuh"

namespace ninfer::ops::detail {
namespace {

using Gemv      = Q8A16GemvSchedule<8, 1, 2, 2560, 8>;
using Sliced8   = Q8A16SlicedKMmaSchedule<8, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 2560>;
using Grouped16 = Q8A16GroupedSlicedKMmaSchedule<16, 2, 2, 2, 1, 2560, Cache::cg, Cache::ca, true>;
using Sliced24  = Q8A16SlicedKMmaSchedule<24, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 2560>;
using Sliced32  = Q8A16SlicedKMmaSchedule<32, 4, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 2560>;
using Sliced40  = Q8A16SlicedKMmaSchedule<40, 4, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 2560>;
using Mma48     = Q8A16MmaSchedule<64, 48, 128, 16, 24, 1, 2>;
using Mma64     = Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>;
using Mma64K128 = Q8A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>;
using Mma96     = Q8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>;
using Mma128 = Q8A16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Q8MmaFragmentPipeline::Serial, Cache::cg,
                                Cache::cg, Cache::ca, false, 2560, 20>;
using Mma64Bulk = Q8A16MmaSchedule<64, 64, 64, 32, 16, 2, 3, Q8MmaFragmentPipeline::Serial,
                                   Cache::cg, Cache::cg, Cache::ca, false, 2560, 20>;

template <class Routes>
auto select_route(int tokens, const Routes& routes) {
    if (tokens <= 1) return routes.template gemv<Gemv>();
    if (tokens <= 8) return routes.template sliced<Sliced8>();
    if (tokens <= 16) return routes.template grouped<Grouped16>();
    if (tokens <= 24) return routes.template sliced<Sliced24>();
    if (tokens <= 32) return routes.template sliced<Sliced32>();
    if (tokens <= 40) return routes.template sliced<Sliced40>();
    if (tokens <= 48) return routes.template mma<Mma48>();
    if (tokens <= 63) return routes.template mma<Mma64>();
    if (tokens <= 64) return routes.template mma<Mma64K128>();
    if (tokens <= 96) return routes.template mma<Mma96>();
    if (tokens <= 128) return routes.template mma<Mma128>();
    if (tokens <= 512) return routes.template mma<Mma64Bulk>();
    return routes.template mma<Mma128>();
}

} // namespace

void q8_gdn_projection_n16384_k2560(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    cudaStream_t stream) {
    using Output = LinearBf16SegmentedOutput<10240, 6144>;
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    select_route(x.ne[1], Q8A16OutputLaunch<Output>{q8_linear_operands(x, weight), output, stream});
}
} // namespace ninfer::ops::detail
