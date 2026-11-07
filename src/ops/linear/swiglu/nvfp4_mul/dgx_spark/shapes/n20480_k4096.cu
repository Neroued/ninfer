#include "ops/linear/swiglu/nvfp4_mul/dispatch.h"
#include "ops/linear/swiglu/nvfp4_mul/launch.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_instances.cuh"

namespace ninfer::ops::detail {
namespace {
using Route0 = Nvfp4ScheduleInstance<
    Nvfp4A16SimtSchedule<4, 2, 2, 16, 2, 4, Nvfp4SimtActivationAccess::TokenPacked,
                         Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 1,
                         Nvfp4SimtBlockOrder::RowsContiguous, 2>,
    4096, 2>;
using Route1 = Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 8, 2>, 4096>;
using Route2 = Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 1>, 4096>;
using Route3 = Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 64, 128, 16, 32, 2, 2, Cache::cg, Cache::cg,
                        Nvfp4MmaFragmentPipeline::PingPong, true, 0, true>, 4096>;
using Route4 = Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 128, 64, 16, 64, 2, 2, Cache::cg, Cache::cg,
                        Nvfp4MmaFragmentPipeline::PingPong, true, 0, true>, 4096>;
using Route5 =
    Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<128, 128, 64, 32, 64, 2, 2, Cache::cg, Cache::cg,
                                              Nvfp4MmaFragmentPipeline::PingPong, true, 8>,
                          4096>;

template <class Product>
void launch(const Tensor& x, const Weight& weight, Tensor& out, Product product,
            cudaStream_t stream) {
    const auto operands = nvfp4_mul_a16_operands(x, weight);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), weight.n / 2};
    const int tokens = operands.tokens;
    if (tokens <= 2) return nvfp4_mul_swiglu_simt<Route0>(operands, output, product, stream);
    if (tokens <= 16) return nvfp4_mul_swiglu_sliced<Route1>(operands, output, product, stream);
    if (tokens <= 32) return nvfp4_mul_swiglu_sliced<Route2>(operands, output, product, stream);
    if (tokens <= 64) return nvfp4_mul_swiglu_mma<Route3>(operands, output, product, stream);
    if (tokens <= 128) return nvfp4_mul_swiglu_mma<Route4>(operands, output, product, stream);
    nvfp4_mul_swiglu_mma<Route5>(operands, output, product, stream);
}
} // namespace

void launch_nvfp4_mul_swiglu_n20480_k4096(const Tensor& x, const Weight& weight, const Tensor*,
                                          Tensor& out, float limit, bool clamped,
                                          cudaStream_t stream) {
    if (clamped)
        launch(x, weight, out, ClampedSwiGluProduct{limit}, stream);
    else
        launch(x, weight, out, SwiGluProduct{}, stream);
}
} // namespace ninfer::ops::detail
