#include "ops/linear/swiglu/nvfp4/dispatch.h"
#include "ops/linear/kernels/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_instances.cuh"
#include "ops/linear/swiglu/product_epilogue.cuh"
#include "core/layout.h"

namespace ninfer::ops::detail {
namespace {
template <class Product>
void launch(const Tensor& x, const Weight& w, Tensor& y, Product product, LinearPolicy policy,
            MemoryArena& ws, cudaStream_t stream) {
    const int t  = x.ne[1];
    const auto p = nvfp4_a16_operands(x, w);
    const LinearBf16Output out{static_cast<__nv_bfloat16*>(y.data), 6144};
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
        if (policy != LinearPolicy::AllowA4)
            throw std::invalid_argument("NVFP4 SwiGLU A16 supports T<=16");
        auto scope   = ws.scope();
        auto scratch = allocate_nvfp4_a4_workspace(ws, t, 4096);
        const auto layout =
            t >= 129 ? ((t <= 256) ? Nvfp4ScaleLayout::Tiled128 : Nvfp4ScaleLayout::Tiled256)
                     : Nvfp4ScaleLayout::RowMajor;
        launch_nvfp4_a4_quantize(x, w, scratch, layout, stream);
        const auto a4 = nvfp4_a4_operands(w, scratch, t, layout);
        if (t >= 129) {
            constexpr bool clamped = std::is_same_v<Product, ClampedSwiGluProduct>;
            float limit            = 0;
            if constexpr (clamped) limit = product.limit;
            launch_nvfp4_swiglu_n12288_k4096_tma(a4, y.data, limit, clamped, stream);
        } else if (t <= 32) {
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

std::size_t nvfp4_swiglu_n12288_k4096_capacity(LinearPolicy policy, int lo, int hi) {
    if (lo <= 0 || hi < lo || !valid_linear_policy(policy))
        throw std::invalid_argument("NVFP4 SwiGLU invalid interval/policy");
    if (policy != LinearPolicy::AllowA4 && hi > 16)
        throw std::invalid_argument("NVFP4 SwiGLU A16 supports T<=16");
    return hi <= 16 ? 0 : nvfp4_a4_workspace_capacity_bytes(hi, 4096);
}

void launch_nvfp4_swiglu_n12288_k4096(const Tensor& x, const Weight& w, Tensor& y, float limit,
                                      bool clamped, LinearPolicy policy, MemoryArena& ws,
                                      cudaStream_t stream) {
    (void)nvfp4_swiglu_n12288_k4096_capacity(policy, x.ne[1], x.ne[1]);
    if (clamped)
        launch(x, w, y, ClampedSwiGluProduct{limit}, policy, ws, stream);
    else
        launch(x, w, y, SwiGluProduct{}, policy, ws, stream);
}
} // namespace ninfer::ops::detail
