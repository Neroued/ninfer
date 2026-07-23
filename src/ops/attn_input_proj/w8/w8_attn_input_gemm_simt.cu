#include "ops/attn_input_proj/w8/w8_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/w8/w8_rowsplit_gemm_simt.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kTargetRows    = 9216;
constexpr int kCompanionRows = 6144;
constexpr int kHidden        = 2048;
constexpr int kRowsPerCta    = 8;
constexpr int kStages        = 2;
constexpr int kCols          = 4;
using TargetOutput           = W8SplitOutput4<4096, 512, 4096, 512>;
using CompanionOutput        = W8SplitOutput3<4096, 1024, 1024>;

template <W8KernelVariant Variant, int Rows, class Output>
void launch_variant(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    const dim3 grid(Rows / kRowsPerCta, static_cast<unsigned>(div_up(x.ne[1], kCols)), 1u);
    w8_rowsplit_gemm_simt_kernel<W8RowSplitSimtSchedule, kCols, kRowsPerCta, kStages, Variant,
                                 W8Epilogue::Store, Output><<<grid, kRowsPerCta * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output, Rows, kHidden, x.ne[1], kHidden,
        kHidden / 1024);
}

template <int Rows, class Output>
void dispatch_variant(W8KernelVariant variant, const Tensor& x, const Weight& weight, Output output,
                      cudaStream_t stream) {
    if (variant == W8KernelVariant::Full) {
        launch_variant<W8KernelVariant::Full, Rows>(x, weight, output, stream);
    } else if (variant == W8KernelVariant::Predicated) {
        launch_variant<W8KernelVariant::Predicated, Rows>(x, weight, output, stream);
    } else {
        throw std::invalid_argument("W8 attention input SIMT requires a tiled variant");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void w8_attn_input_simt_r8_c4_launch(W8KernelVariant variant, const Tensor& x, const Weight& weight,
                                     Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                     cudaStream_t stream) {
    static_assert((4096 % kRowsPerCta) == 0 && (512 % kRowsPerCta) == 0);
    const TargetOutput output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    dispatch_variant<kTargetRows>(variant, x, weight, output, stream);
}

void w8_attn_input_simt_r8_c4_launch(W8KernelVariant variant, const Tensor& x, const Weight& weight,
                                     Tensor& q, Tensor& k, Tensor& v, cudaStream_t stream) {
    static_assert((4096 % kRowsPerCta) == 0 && (1024 % kRowsPerCta) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    dispatch_variant<kCompanionRows>(variant, x, weight, output, stream);
}

} // namespace ninfer::ops::detail
