#include "ops/linear/swiglu/nvfp4/dispatch.h"
#include "ops/linear/kernels/nvfp4/nvfp4_a4_tma.cuh"
#include "ops/linear/swiglu/product_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
template <int Tokens, int Stages, class Product>
void launch(const Nvfp4A4Operands& p, void* out, Product product, cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<Tokens, Stages, 1>, 4096>;
    launch_nvfp4_a4_tma_mma<S>(p, LinearBf16Output{static_cast<__nv_bfloat16*>(out), 6144},
                               BasicSwiGluTokenMajorMmaEpilogue<Product>{product}, stream,
                               SwiGluTokenMajorMmaRows<S>{});
}

template <class Product>
void dispatch(const Nvfp4A4Operands& p, void* out, Product product, cudaStream_t stream) {
    if (p.scale_layout == Nvfp4ScaleLayout::Tiled128)
        launch<128, 4>(p, out, product, stream);
    else
        launch<256, 3>(p, out, product, stream);
}
} // namespace

void launch_nvfp4_swiglu_n12288_k4096_tma(const Nvfp4A4Operands& p, void* out, float limit,
                                          bool clamped, cudaStream_t stream) {
    if (clamped)
        dispatch(p, out, ClampedSwiGluProduct{limit}, stream);
    else
        dispatch(p, out, SwiGluProduct{}, stream);
}
} // namespace ninfer::ops::detail
