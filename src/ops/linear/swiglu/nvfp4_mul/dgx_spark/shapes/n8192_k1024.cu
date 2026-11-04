#include "ops/linear/swiglu/nvfp4_mul/dispatch.h"
#include "ops/linear/swiglu/nvfp4_mul/launch.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_instances.cuh"

namespace ninfer::ops::detail {
namespace {
using Route0 = Nvfp4ScheduleInstance<
    Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 2>, 1024>;
using Route1 = Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 2, 2>, 1024>;
using Route2 = Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 2, 2>, 1024>;
using Route3 = Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>, 1024>;
using Route4 =
    Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<128, 128, 64, 64, 32, 2, 2, Cache::cg, Cache::cg,
                                              Nvfp4MmaFragmentPipeline::PingPong, true, 8>,
                          1024>;

template <class Product>
void launch(const Tensor& x, const Weight& weight, Tensor& out, Product product,
            cudaStream_t stream) {
    const auto operands = nvfp4_mul_a16_operands(x, weight);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), weight.n / 2};
    const int tokens = operands.tokens;
    if (tokens <= 1) return nvfp4_mul_swiglu_gemv<Route0>(operands, output, product, stream);
    if (tokens <= 16) return nvfp4_mul_swiglu_sliced<Route1>(operands, output, product, stream);
    if (tokens <= 32) return nvfp4_mul_swiglu_sliced<Route2>(operands, output, product, stream);
    if (tokens <= 64) return nvfp4_mul_swiglu_mma<Route3>(operands, output, product, stream);
    nvfp4_mul_swiglu_mma<Route4>(operands, output, product, stream);
}
} // namespace

void launch_nvfp4_mul_swiglu_n8192_k1024(const Tensor& x, const Weight& weight, const Tensor* bias,
                                         Tensor& out, float limit, bool clamped,
                                         cudaStream_t stream) {
    if (bias != nullptr) {
        launch(x, weight, out,
               BiasedClampedSwiGluProduct{static_cast<const __nv_bfloat16*>(bias->data),
                                          weight.n / 2, limit},
               stream);
        return;
    }
    if (clamped)
        launch(x, weight, out, ClampedSwiGluProduct{limit}, stream);
    else
        launch(x, weight, out, SwiGluProduct{}, stream);
}
} // namespace ninfer::ops::detail
