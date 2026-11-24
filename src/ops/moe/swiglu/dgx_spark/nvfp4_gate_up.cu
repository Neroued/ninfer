#include "ops/moe/swiglu/dispatch.h"
#include "ops/moe/swiglu/kernel/output.cuh"
#include "ops/moe/swiglu/quantized_product.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_instances.cuh"
#include "ops/linear/swiglu/product_epilogue.cuh"

namespace ninfer::ops::detail::swiglu_ffn {
namespace {
template <class Product, class Output>
void launch(const Tensor& x, const Weight& w, Output out, Product product, MemoryArena& ws,
            cudaStream_t stream) {
    const int t  = x.ne[1];
    const auto p = nvfp4_a16_operands(x, w);
    const BasicSwiGluPairEpilogue<Product> epilogue{product};
    if (t == 1) {
        using S = Nvfp4ScheduleInstance<
            Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 2>,
            4096>;
        launch_nvfp4_a16_gemv<S>(p, out, epilogue, stream, SwiGluBranchRows<1>{});
    } else if (t <= 2) {
        using Schedule =
            Nvfp4A16SimtSchedule<8, 1, 2, 16, 2, 1, Nvfp4SimtActivationAccess::SharedPhase,
                                 Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                 Nvfp4SimtBlockOrder::RowsContiguous, 1>;
        launch_nvfp4_a16_simt<Nvfp4ScheduleInstance<Schedule, 4096, 2, true>>(
            p, out, epilogue, stream, SwiGluBranchRows<1>{});
    } else if (t <= 16) {
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 8, 2>, 4096>>(
            p, out, epilogue, stream, SwiGluBranchRows<8>{});
    } else {
        auto scope            = ws.scope();
        auto scratch          = allocate_nvfp4_a4_workspace(ws, t, 4096);
        constexpr auto layout = Nvfp4ScaleLayout::RowMajor;
        launch_nvfp4_a4_quantize(x, w, scratch, layout, stream);
        const auto a4 = nvfp4_a4_operands(w, scratch, t, layout);
        if (t <= 32) {
            using S = Nvfp4ScheduleInstance<Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>, 4096>;
            launch_nvfp4_a4_mma<S>(a4, out, BasicSwiGluTokenMajorMmaEpilogue<Product>{product},
                                   stream, SwiGluTokenMajorMmaRows<S>{});
        } else if (t <= 64) {
            using S = Nvfp4ScheduleInstance<Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>, 4096>;
            launch_nvfp4_a4_mma<S>(a4, out, BasicSwiGluTokenMajorMmaEpilogue<Product>{product},
                                   stream, SwiGluTokenMajorMmaRows<S>{});
        } else {
            using S = Nvfp4ScheduleInstance<Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>, 4096>;
            launch_nvfp4_a4_mma<S>(a4, out, BasicSwiGluTokenMajorMmaEpilogue<Product>{product},
                                   stream, SwiGluTokenMajorMmaRows<S>{});
        }
    }
}
} // namespace

void launch_nvfp4_gu_float(const Tensor& x, const Weight& w, float* y, float limit, MemoryArena& ws,
                           cudaStream_t stream) {
    launch(x, w, SwiGluFloatOutput{y, 6144}, ClampedSwiGluProduct{limit}, ws, stream);
}

void launch_nvfp4_gu_quantized(const Tensor& x, const Weight& w, Nvfp4A4Workspace dst,
                               Nvfp4ScaleLayout down_layout, float down_divisor, float limit,
                               MemoryArena& ws, cudaStream_t stream) {
    const int t       = x.ne[1];
    auto scope        = ws.scope();
    const auto src    = allocate_nvfp4_a4_workspace(ws, t, 4096);
    const auto layout = t <= 256 ? Nvfp4ScaleLayout::Tiled128 : Nvfp4ScaleLayout::Tiled256;
    launch_nvfp4_a4_quantize(x, w, src, layout, stream);
    const auto p = nvfp4_a4_operands(w, src, t, layout);
    const QuantizedProductOutput out{
        dst.codes,   dst.scales, down_divisor,
        down_layout, t,          nvfp4_a4_padded_tokens(t, down_layout)};
    launch_nvfp4_gu_quantized_tma(p, out, limit, stream);
}
} // namespace ninfer::ops::detail::swiglu_ffn
