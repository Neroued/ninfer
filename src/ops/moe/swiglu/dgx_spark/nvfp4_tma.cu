#include "ops/moe/swiglu/dispatch.h"
#include "ops/moe/swiglu/quantized_product.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_a4_tma.cuh"
#include "ops/linear/swiglu/token_major_mma_epilogue.cuh"

namespace ninfer::ops::detail::swiglu_ffn {
namespace {
template <int Tokens, int Stages>
void launch(const Nvfp4A4Operands& p, QuantizedProductOutput out, float limit,
            cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<Tokens, Stages, 1>, 4096>;
    launch_nvfp4_a4_tma_mma<S>(p, out, QuantizedProductEpilogue<ClampedSwiGluProduct>{{limit}},
                               stream, SwiGluTokenMajorMmaRows<S>{});
}
} // namespace

void launch_nvfp4_gu_quantized_tma(const Nvfp4A4Operands& p, const QuantizedProductOutput& out,
                                   float limit, cudaStream_t stream) {
    if (p.scale_layout == Nvfp4ScaleLayout::Tiled128)
        launch<128, 4>(p, out, limit, stream);
    else
        launch<256, 3>(p, out, limit, stream);
}

namespace {
template <int Tokens, int Stages, Nvfp4ScaleLayout Layout>
void down(const Weight& w, Tensor& out, Nvfp4A4Workspace scratch, int tokens, cudaStream_t stream) {
    launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<Tokens, Stages, 1>, 6144>>(
        nvfp4_a4_operands(w, scratch, tokens, Layout),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, LinearIdentityEpilogue{},
        stream);
}
} // namespace

void launch_nvfp4_down_tma128(const Weight& w, Tensor& out, Nvfp4A4Workspace scratch, int tokens,
                              cudaStream_t stream) {
    down<128, 4, Nvfp4ScaleLayout::Tiled128>(w, out, scratch, tokens, stream);
}

void launch_nvfp4_down_tma256(const Weight& w, Tensor& out, Nvfp4A4Workspace scratch, int tokens,
                              cudaStream_t stream) {
    down<256, 3, Nvfp4ScaleLayout::Tiled256>(w, out, scratch, tokens, stream);
}
} // namespace ninfer::ops::detail::swiglu_ffn
